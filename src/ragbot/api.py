"""FastAPI service: POST /ask, GET /health. Docs at /docs."""
from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import get_config

app = FastAPI(title="RAG Chatbot", version="1.0.0")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)


@lru_cache(maxsize=1)
def _engine():
    from .engine import RAGEngine

    return RAGEngine(get_config())


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "provider": get_config().provider}


@app.post("/ask")
def ask(req: AskRequest) -> dict:
    try:
        return _engine().ask(req.question).to_dict()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
