"""Stage 3 server — validation guardrail with one retry and force_bad demo knob.

Run: uvicorn serve_stage3:app --port 8000 --reload
"""

from dotenv import find_dotenv, load_dotenv
from fastapi import FastAPI, HTTPException
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError

load_dotenv(find_dotenv())

app = FastAPI()
client = OpenAI()


class Answer(BaseModel):
    answer: str
    confidence: float = Field(ge=0.0, le=1.0)
    sources_needed: bool


class AskRequest(BaseModel):
    question: str
    force_bad: bool = False


class AskResponse(BaseModel):
    answer: Answer
    tokens_used: int


def call_structured(question: str) -> tuple[Answer, int]:
    completion = client.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": question}],
        response_format=Answer,
    )
    parsed = completion.choices[0].message.parsed
    if parsed is None:
        raise ValueError("Model returned no parseable structured output")
    tokens_used = completion.usage.total_tokens if completion.usage else 0
    return parsed, tokens_used


def call_unsafe(question: str) -> tuple[Answer, int]:
    """Demo path: bad instruction makes confidence a string so validation fails."""
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
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
    answer = Answer.model_validate_json(raw)
    tokens_used = completion.usage.total_tokens if completion.usage else 0
    return answer, tokens_used


@app.post("/ask")
def ask(body: AskRequest) -> AskResponse:
    last_error: str | None = None

    for attempt in range(2):
        try:
            if body.force_bad and attempt == 0:
                answer, tokens_used = call_unsafe(body.question)
            else:
                answer, tokens_used = call_structured(body.question)
            return AskResponse(answer=answer, tokens_used=tokens_used)
        except (ValidationError, ValueError) as exc:
            last_error = str(exc)
            continue

    raise HTTPException(
        status_code=502,
        detail=f"Model response failed schema validation after retry: {last_error}",
    )
