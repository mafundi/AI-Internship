"""Stage 2 server — structured output turns the chatbot into a component.

Run: uvicorn serve_stage2:app --port 8000 --reload
"""

from dotenv import find_dotenv, load_dotenv
from fastapi import FastAPI
from openai import OpenAI
from pydantic import BaseModel, Field

load_dotenv(find_dotenv())

app = FastAPI()
client = OpenAI()


class Answer(BaseModel):
    """Structured model output — this is what turns a chatbot into a component."""

    answer: str
    confidence: float = Field(ge=0.0, le=1.0)
    sources_needed: bool


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: Answer
    tokens_used: int


@app.post("/ask")
def ask(body: AskRequest) -> AskResponse:
    """Force the model into a fixed schema via OpenAI structured output."""

    completion = client.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": body.question}],
        response_format=Answer,
    )

    parsed = completion.choices[0].message.parsed
    if parsed is None:
        raise ValueError("Model returned no parseable structured output")

    tokens_used = completion.usage.total_tokens if completion.usage else 0
    return AskResponse(answer=parsed, tokens_used=tokens_used)
