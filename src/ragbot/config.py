"""Central configuration, driven by environment variables (.env supported)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # python-dotenv is optional at import time
    pass


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, default))


def _float(name: str, default: float) -> float:
    return float(os.getenv(name, default))


@dataclass(frozen=True)
class AppConfig:
    provider: str = "openai"  # "openai" | "bedrock"

    # OpenAI
    openai_llm_model: str = "gpt-4o-mini"
    openai_embed_model: str = "text-embedding-3-small"

    # AWS Bedrock
    aws_region: str = "us-east-1"
    bedrock_llm_model: str = "anthropic.claude-3-5-sonnet-20240620-v1:0"
    bedrock_embed_model: str = "amazon.titan-embed-text-v2:0"

    # Paths
    data_dir: Path = Path("data/pdfs")
    index_dir: Path = Path("storage/index")

    # Chunking / retrieval
    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k: int = 4
    similarity_cutoff: float = 0.30
    embed_batch_size: int = 64
    cache_size: int = 256

    def validate(self) -> None:
        if self.provider not in {"openai", "bedrock"}:
            raise ValueError(f"RAG_PROVIDER must be 'openai' or 'bedrock', got {self.provider!r}")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("RAG_CHUNK_OVERLAP must be smaller than RAG_CHUNK_SIZE")
        if self.top_k < 1:
            raise ValueError("RAG_TOP_K must be >= 1")


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    cfg = AppConfig(
        provider=os.getenv("RAG_PROVIDER", "openai").lower(),
        openai_llm_model=os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini"),
        openai_embed_model=os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small"),
        aws_region=os.getenv("AWS_REGION", "us-east-1"),
        bedrock_llm_model=os.getenv(
            "BEDROCK_LLM_MODEL", "anthropic.claude-3-5-sonnet-20240620-v1:0"
        ),
        bedrock_embed_model=os.getenv("BEDROCK_EMBED_MODEL", "amazon.titan-embed-text-v2:0"),
        data_dir=Path(os.getenv("RAG_DATA_DIR", "data/pdfs")),
        index_dir=Path(os.getenv("RAG_INDEX_DIR", "storage/index")),
        chunk_size=_int("RAG_CHUNK_SIZE", 512),
        chunk_overlap=_int("RAG_CHUNK_OVERLAP", 64),
        top_k=_int("RAG_TOP_K", 4),
        similarity_cutoff=_float("RAG_SIMILARITY_CUTOFF", 0.30),
        embed_batch_size=_int("RAG_EMBED_BATCH_SIZE", 64),
        cache_size=_int("RAG_CACHE_SIZE", 256),
    )
    cfg.validate()
    return cfg
