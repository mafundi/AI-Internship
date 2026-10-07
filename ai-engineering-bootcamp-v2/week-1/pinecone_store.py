"""Pinecone vector store — embeddings and index access for Session 2 RAG.

Uses the same embedding model at ingest and query time.
Secrets come only from environment variables, never from this file.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from pinecone import Pinecone

load_dotenv(Path(__file__).resolve().parent / ".env")
load_dotenv()

# Must match at ingest and query. text-embedding-3-small is 1536 dimensions.
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIMENSIONS = 1536
DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 100

_openai: OpenAI | None = None
_pinecone: Pinecone | None = None


def _require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing environment variable: {name}")
    return value


def openai_client() -> OpenAI:
    global _openai
    if _openai is None:
        _require_env("OPENAI_API_KEY")
        _openai = OpenAI()
    return _openai


def pinecone_client() -> Pinecone:
    global _pinecone
    if _pinecone is None:
        _pinecone = Pinecone(api_key=_require_env("PINECONE_API_KEY"))
    return _pinecone


def index_name() -> str:
    return _require_env("PINECONE_INDEX_NAME")


def namespace() -> str | None:
    value = os.getenv("PINECONE_NAMESPACE", "").strip()
    return value or None


def get_index():
    return pinecone_client().Index(index_name())


def chunk_size() -> int:
    raw = os.getenv("INGEST_CHUNK_SIZE", str(DEFAULT_CHUNK_SIZE)).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError("INGEST_CHUNK_SIZE must be an integer") from exc
    if value < 1:
        raise RuntimeError("INGEST_CHUNK_SIZE must be >= 1")
    return value


def chunk_overlap() -> int:
    raw = os.getenv("INGEST_CHUNK_OVERLAP", str(DEFAULT_CHUNK_OVERLAP)).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError("INGEST_CHUNK_OVERLAP must be an integer") from exc
    if value < 0:
        raise RuntimeError("INGEST_CHUNK_OVERLAP must be >= 0")
    size = chunk_size()
    if value >= size:
        raise RuntimeError("INGEST_CHUNK_OVERLAP must be smaller than INGEST_CHUNK_SIZE")
    return value


def split_text(text: str) -> list[str]:
    """Split document text with RecursiveCharacterTextSplitter."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size(),
        chunk_overlap=chunk_overlap(),
        separators=["\n\n", "\n", " ", ""],
    )
    return [chunk.strip() for chunk in splitter.split_text(text) if chunk.strip()]


def _pinecone_safe_metadata(values: dict[str, Any] | None) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    if not values:
        return safe
    for key, value in values.items():
        if isinstance(value, (str, int, float, bool)):
            safe[str(key)] = value
        elif value is None:
            continue
        else:
            safe[str(key)] = str(value)
    return safe


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Turn strings into vectors with the ingest/query embedding model."""

    if not texts:
        return []
    response = openai_client().embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in sorted(response.data, key=lambda row: row.index)]


def upsert_texts(
    ids: list[str],
    texts: list[str],
    metadatas: list[dict[str, Any]] | None = None,
) -> int:
    """Embed chunks and write them to Pinecone. Same model as query_similar()."""

    if len(ids) != len(texts):
        raise ValueError("ids and texts must be the same length")
    if metadatas is not None and len(metadatas) != len(texts):
        raise ValueError("metadatas must match texts")

    records = []
    embed_batch = 64
    for start in range(0, len(texts), embed_batch):
        batch_texts = texts[start : start + embed_batch]
        vectors = embed_texts(batch_texts)
        for offset, (doc_id, values, text) in enumerate(
            zip(ids[start : start + embed_batch], vectors, batch_texts)
        ):
            meta = dict(metadatas[start + offset]) if metadatas else {}
            meta["text"] = text
            meta["embedding_model"] = EMBEDDING_MODEL
            records.append({"id": doc_id, "values": values, "metadata": meta})

    index = get_index()
    ns = namespace()
    upsert_batch = 100
    for start in range(0, len(records), upsert_batch):
        kwargs: dict[str, Any] = {"vectors": records[start : start + upsert_batch]}
        if ns:
            kwargs["namespace"] = ns
        index.upsert(**kwargs)
    return len(records)


def ingest_document(
    document_id: str,
    text: str,
    source: str = "",
    extra_metadata: dict[str, Any] | None = None,
) -> int:
    """Chunk, embed with text-embedding-3-small, and upsert into Pinecone."""

    chunks = split_text(text)
    if not chunks:
        return 0

    extra = _pinecone_safe_metadata(extra_metadata)
    for reserved in ("document_id", "chunk_index", "source", "text"):
        extra.pop(reserved, None)

    ids = [f"{document_id}:{index}" for index in range(len(chunks))]
    metadatas = [
        {
            **extra,
            "document_id": document_id,
            "chunk_index": index,
            "source": source,
        }
        for index in range(len(chunks))
    ]
    return upsert_texts(ids, chunks, metadatas)


def query_similar(question: str, top_k: int = 5) -> list[dict[str, Any]]:
    """Embed the question with the same model used at ingest, then search."""

    vector = embed_texts([question])[0]
    kwargs: dict[str, Any] = {
        "vector": vector,
        "top_k": top_k,
        "include_metadata": True,
    }
    ns = namespace()
    if ns:
        kwargs["namespace"] = ns
    result = get_index().query(**kwargs)
    matches = []
    for match in result.matches or []:
        matches.append(
            {
                "id": match.id,
                "score": match.score,
                "text": (match.metadata or {}).get("text"),
                "metadata": dict(match.metadata or {}),
            }
        )
    return matches


def pinecone_health() -> dict[str, Any]:
    """Reach Pinecone and report index stats. Does not print secrets."""

    missing = [
        name
        for name in ("OPENAI_API_KEY", "PINECONE_API_KEY", "PINECONE_INDEX_NAME")
        if not os.getenv(name, "").strip()
    ]
    report: dict[str, Any] = {
        "ok": False,
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dimensions": EMBEDDING_DIMENSIONS,
        "openai_key_set": bool(os.getenv("OPENAI_API_KEY", "").strip()),
        "pinecone_key_set": bool(os.getenv("PINECONE_API_KEY", "").strip()),
        "index_name_set": bool(os.getenv("PINECONE_INDEX_NAME", "").strip()),
        "namespace": namespace(),
    }
    if missing:
        report["error"] = f"Missing environment variable(s): {', '.join(missing)}"
        return report

    try:
        stats = get_index().describe_index_stats()
        stats_map = stats if isinstance(stats, dict) else getattr(stats, "__dict__", {})
        report["ok"] = True
        report["pinecone_reachable"] = True
        report["index"] = index_name()
        report["vector_count"] = getattr(stats, "total_vector_count", None) or stats_map.get(
            "total_vector_count"
        )
        report["index_dimension"] = getattr(stats, "dimension", None) or stats_map.get("dimension")
        return report
    except Exception as exc:
        report["pinecone_reachable"] = False
        report["error"] = str(exc)
        return report
