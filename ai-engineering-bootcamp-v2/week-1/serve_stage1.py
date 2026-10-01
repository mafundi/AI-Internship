"""Stage 1 server — bare /ask with typed I/O and real token usage.

Run: uvicorn serve_stage1:app --port 8000 --reload
"""

from dotenv import find_dotenv, load_dotenv
from fastapi import FastAPI
from openai import OpenAI
from pydantic import BaseModel

load_dotenv(find_dotenv())

app = FastAPI()
client = OpenAI()


class AskRequest(BaseModel):
    """Typed request body so bad input is rejected before we spend tokens."""

    question: str


class AskResponse(BaseModel):
    """Typed response so callers always get the same shape back."""

    answer: str
    tokens_used: int


@app.post("/ask")
def ask(body: AskRequest) -> AskResponse:
    """Answer one question and surface real token usage for cost visibility."""

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": body.question}],
    )

    answer = completion.choices[0].message.content or ""
    tokens_used = completion.usage.total_tokens if completion.usage else 0

    return AskResponse(answer=answer, tokens_used=tokens_used)
