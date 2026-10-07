"""Week 1 live demo — five stages in one file, built up live in class."""

import os
import re
import time

from dotenv import find_dotenv, load_dotenv
from fastapi import FastAPI, HTTPException, Query
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError

from pinecone_store import EMBEDDING_MODEL, ingest_document, pinecone_health, query_similar

# Load .env from this folder, or walk up to the repo root if that is where it lives.
load_dotenv(find_dotenv())

# Reuse one client so TLS handshakes are not repeated on every request.
app = FastAPI()
client = OpenAI()  # Reads OPENAI_API_KEY from the environment; never hardcode keys.

# Stage 4 default — strong general model; swap at request time for the live demo.
DEFAULT_MODEL = "gpt-4o"

# Stage 5 — per-1K-token input/output USD (derived from OpenAI list prices).
MODEL_PRICES_PER_1K: dict[str, tuple[float, float]] = {
    "gpt-4o": (0.0025, 0.01),
    "gpt-4o-mini": (0.00015, 0.0006),
    "o3-mini": (0.0011, 0.0044),
}

DEFAULT_ASK_TOP_K = 5

GROUNDING_PROMPT_TEMPLATE = """Answer using ONLY the context below.
If the context does not contain the answer, say:
"I don't have enough information to answer that."
Cite the document_id of each chunk you used.

Context:
{retrieved_chunks}

Question: {question}
"""


class Answer(BaseModel):
    """Structured model output — this is what turns a chatbot into a component."""

    answer: str
    confidence: float = Field(ge=0.0, le=1.0)
    sources_needed: bool
    citations: list[str] = Field(
        default_factory=list,
        description=(
            "document_id values of chunks you actually used, e.g. ['POL-101']. "
            "Empty list if you refused because the context was insufficient."
        ),
    )


class AskRequest(BaseModel):
    """Typed request body so bad input is rejected before we spend tokens."""

    question: str
    force_bad: bool = False  # Stage 3 demo knob — first attempt breaks schema on purpose.
    model: str | None = None  # Stage 4 — optional override to swap models live.
    top_k: int | None = Field(default=None, ge=1, le=20)  # RAG chunk count; default ASK_TOP_K / 5.


class AskResponse(BaseModel):
    """Typed response so callers always get the same shape back."""

    answer: Answer
    tokens_used: int
    model: str
    latency_ms: int
    cost_usd: float
    retrieved_chunk_ids: list[str] = []


class IngestRequest(BaseModel):
    """Document text to chunk, embed, and store in Pinecone."""

    text: str
    document_id: str
    source: str | None = None  # optional filename or URL
    metadata: dict | None = None


class IngestResponse(BaseModel):
    document_id: str
    chunks_indexed: int
    status: str


class RetrieveMatch(BaseModel):
    id: str
    score: float | None = None
    document_id: str | None = None
    chunk_index: int | str | None = None
    source: str | None = None
    text: str | None = None


class RetrieveResponse(BaseModel):
    q: str
    embedding_model: str
    top_k: int
    matches: list[RetrieveMatch]


def ask_top_k(requested: int | None) -> int:
    if requested is not None:
        return requested
    raw = os.getenv("ASK_TOP_K", str(DEFAULT_ASK_TOP_K)).strip()
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_ASK_TOP_K
    return max(1, min(20, value))


def format_context_chunks(matches: list[dict]) -> str:
    blocks = []
    for match in matches:
        meta = match.get("metadata") or {}
        document_id = meta.get("document_id") or "unknown"
        text = match.get("text") or meta.get("text") or ""
        blocks.append(f"[{document_id}] {text}")
    return "\n\n".join(blocks)


REFUSAL_ANSWER = "I don't have enough information to answer that."


def retrieved_document_ids(matches: list[dict]) -> list[str]:
    """Stable unique document_ids from retrieved chunks, retrieval order."""

    seen: list[str] = []
    for match in matches:
        meta = match.get("metadata") or {}
        document_id = str(meta.get("document_id") or "").strip()
        if document_id and document_id not in seen:
            seen.append(document_id)
    return seen


def apply_citations(answer: Answer, matches: list[dict]) -> Answer:
    """Keep only real retrieved document_ids and put [DOC-ID] in the answer text."""

    allowed = retrieved_document_ids(matches)
    allowed_set = set(allowed)
    from_field = [item.strip() for item in answer.citations if item and item.strip() in allowed_set]
    from_text = [
        token.strip("[]")
        for token in re.findall(r"\[[A-Za-z0-9._-]+\]", answer.answer)
        if token.strip("[]") in allowed_set
    ]

    merged: list[str] = []
    for document_id in from_field + from_text:
        if document_id not in merged:
            merged.append(document_id)

    text = answer.answer.strip()
    is_refusal = text.strip('"') == REFUSAL_ANSWER
    if is_refusal:
        merged = []
    elif not merged:
        merged = list(allowed)

    if merged:
        missing = [document_id for document_id in merged if f"[{document_id}]" not in text]
        if missing:
            text = f"{text} " + " ".join(f"[{document_id}]" for document_id in missing)

    return answer.model_copy(update={"answer": text, "citations": merged})


def build_grounded_prompt(question: str, matches: list[dict]) -> str:
    """Fill GROUNDING_PROMPT_TEMPLATE with the user question and retrieved chunks."""

    return GROUNDING_PROMPT_TEMPLATE.format(
        question=question.strip(),
        retrieved_chunks=format_context_chunks(matches),
    )


def compute_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Turn real usage into dollars — same prompt, different model, different cost."""

    prices = MODEL_PRICES_PER_1K.get(model, MODEL_PRICES_PER_1K[DEFAULT_MODEL])
    input_per_1k, output_per_1k = prices
    return (prompt_tokens / 1000 * input_per_1k) + (completion_tokens / 1000 * output_per_1k)


def call_model_structured(question: str, model: str) -> tuple[Answer, int, int, int]:
    """
    Stage 2 center: OpenAI structured output forces exactly the Answer schema.
    Returns parsed answer plus token counts from billing metadata.
    """

    completion = client.chat.completions.parse(
        model=model,
        messages=[{"role": "user", "content": question}],
        response_format=Answer,
    )

    parsed = completion.choices[0].message.parsed
    if parsed is None:
        raise ValueError("Model returned no parseable structured output")

    usage = completion.usage
    total = usage.total_tokens if usage else 0
    prompt_tokens = usage.prompt_tokens if usage else 0
    completion_tokens = usage.completion_tokens if usage else 0
    return parsed, total, prompt_tokens, completion_tokens


def call_model_unsafe(question: str, model: str) -> tuple[Answer, int, int, int]:
    """
    Stage 3 demo path: free-form JSON call, then validate locally.
    The bad instruction makes confidence a string so Pydantic rejects it reliably.
    """

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": (
                    f"{question}\n\n"
                    "Reply with ONLY a JSON object using keys answer, confidence, sources_needed. "
                    "Set confidence to the string 'very high' (not a number)."
                ),
            }
        ],
    )

    raw = completion.choices[0].message.content or ""
    # Guardrail: refuse malformed output instead of passing it through to clients.
    answer = Answer.model_validate_json(raw)

    usage = completion.usage
    total = usage.total_tokens if usage else 0
    prompt_tokens = usage.prompt_tokens if usage else 0
    completion_tokens = usage.completion_tokens if usage else 0
    return answer, total, prompt_tokens, completion_tokens


@app.get("/health")
def health() -> dict:
    """Liveness plus Pinecone reachability (no secrets in the response)."""

    pinecone = pinecone_health()
    return {
        "status": "ok" if pinecone.get("ok") else "degraded",
        "service": "ask",
        "pinecone": pinecone,
    }


@app.get("/debug/pinecone")
def debug_pinecone() -> dict:
    """Call this to confirm Pinecone env vars and index access."""

    report = pinecone_health()
    if not report.get("ok"):
        raise HTTPException(status_code=503, detail=report)
    return report


@app.get("/debug/retrieve")
def debug_retrieve(
    q: str = Query(..., min_length=1, description="Question to embed and search"),
    top_k: int = Query(5, ge=1, le=20),
) -> RetrieveResponse:
    """Embed q and return top Pinecone chunks. Does not call the chat LLM.

    curl example:
      curl -s "http://127.0.0.1:8000/debug/retrieve?q=What%20is%20the%20heat%20plan"
    """

    question = q.strip()
    if not question:
        raise HTTPException(status_code=400, detail="q must be a non-empty string")

    try:
        raw_matches = query_similar(question, top_k=top_k)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Retrieve failed: {exc}") from exc

    matches = []
    for row in raw_matches:
        meta = row.get("metadata") or {}
        matches.append(
            RetrieveMatch(
                id=str(row.get("id") or ""),
                score=row.get("score"),
                document_id=meta.get("document_id"),
                chunk_index=meta.get("chunk_index"),
                source=meta.get("source"),
                text=row.get("text") or meta.get("text"),
            )
        )

    return RetrieveResponse(
        q=question,
        embedding_model=EMBEDDING_MODEL,
        top_k=top_k,
        matches=matches,
    )


@app.post("/ingest")
def ingest(body: IngestRequest) -> IngestResponse:
    """Chunk text, embed with text-embedding-3-small, and upsert into Pinecone.

    curl example:
      curl -s -X POST http://127.0.0.1:8000/ingest \\
        -H "Content-Type: application/json" \\
        -d '{"document_id": "heat-plan-1", "source": "heat_plan.txt", "text": "Your document text here..."}'
    """

    document_id = body.document_id.strip()
    text = body.text.strip()
    if not document_id:
        raise HTTPException(status_code=400, detail="document_id must be a non-empty string")
    if not text:
        raise HTTPException(status_code=400, detail="text must be a non-empty string")

    extra = dict(body.metadata or {})
    source = (body.source or extra.get("source") or "").strip()
    extra.pop("source", None)

    try:
        chunks_indexed = ingest_document(
            document_id=document_id,
            text=text,
            source=source,
            extra_metadata=extra,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Ingest failed: {exc}") from exc

    if chunks_indexed == 0:
        raise HTTPException(
            status_code=400,
            detail="text produced no chunks after splitting; send longer non-empty text",
        )

    return IngestResponse(
        document_id=document_id,
        chunks_indexed=chunks_indexed,
        status="ok",
    )


@app.post("/ask")
def ask(body: AskRequest) -> AskResponse:
    """Retrieve top-k chunks, ground the prompt, then run Session 1 generation.

    curl example:
      curl -s -X POST http://127.0.0.1:8000/ask \\
        -H "Content-Type: application/json" \\
        -d '{"question": "What is RAG in one sentence?", "model": "gpt-4o-mini"}'
    """

    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question must be a non-empty string")

    model = body.model or DEFAULT_MODEL
    top_k = ask_top_k(body.top_k)
    start = time.perf_counter()

    try:
        matches = query_similar(question, top_k=top_k)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Retrieve failed: {exc}") from exc

    retrieved_chunk_ids = [str(match.get("id")) for match in matches if match.get("id")]
    grounded_prompt = build_grounded_prompt(question, matches)

    last_error: str | None = None
    # Stage 3: one retry keeps the logic legible while still protecting callers.
    for attempt in range(2):
        try:
            # First attempt with force_bad uses the unsafe path; retry uses structured output.
            use_bad_path = body.force_bad and attempt == 0
            if use_bad_path:
                answer, tokens_used, prompt_tokens, completion_tokens = call_model_unsafe(
                    grounded_prompt, model
                )
            else:
                answer, tokens_used, prompt_tokens, completion_tokens = call_model_structured(
                    grounded_prompt, model
                )

            answer = apply_citations(answer, matches)
            latency_ms = int((time.perf_counter() - start) * 1000)
            cost_usd = compute_cost_usd(model, prompt_tokens, completion_tokens)

            return AskResponse(
                answer=answer,
                tokens_used=tokens_used,
                model=model,
                latency_ms=latency_ms,
                cost_usd=round(cost_usd, 6),
                retrieved_chunk_ids=retrieved_chunk_ids,
            )
        except (ValidationError, ValueError) as exc:
            last_error = str(exc)
            continue

    # Clean failure — never leak a half-parsed response to the client.
    raise HTTPException(
        status_code=502,
        detail=f"Model response failed schema validation after retry: {last_error}",
    )
