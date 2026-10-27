"""PDF ingestion: load -> chunk -> embed -> persist (incremental on re-runs)."""
from __future__ import annotations

import logging
import time

from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.node_parser import SentenceSplitter

from .config import AppConfig
from .llm import configure_models

log = logging.getLogger(__name__)


def load_documents(cfg: AppConfig):
    if not cfg.data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {cfg.data_dir}")
    reader = SimpleDirectoryReader(
        input_dir=str(cfg.data_dir),
        required_exts=[".pdf"],
        recursive=True,
        filename_as_id=True,  # stable ids -> enables incremental refresh
    )
    docs = reader.load_data(show_progress=True)
    if not docs:
        raise RuntimeError(f"No PDFs found in {cfg.data_dir}. Add files and retry.")
    return docs


def build_or_update_index(cfg: AppConfig) -> VectorStoreIndex:
    """Build the index on first run; afterwards only new/changed PDFs are re-embedded."""
    configure_models(cfg)
    start = time.perf_counter()
    docs = load_documents(cfg)
    splitter = SentenceSplitter(chunk_size=cfg.chunk_size, chunk_overlap=cfg.chunk_overlap)

    if (cfg.index_dir / "docstore.json").exists():
        log.info("Existing index found, refreshing changed documents only")
        storage = StorageContext.from_defaults(persist_dir=str(cfg.index_dir))
        index = load_index_from_storage(storage, transformations=[splitter])
        refreshed = index.refresh_ref_docs(docs)
        log.info("Re-embedded %d document(s)", sum(refreshed))
    else:
        index = VectorStoreIndex.from_documents(
            docs, transformations=[splitter], show_progress=True
        )

    cfg.index_dir.mkdir(parents=True, exist_ok=True)
    index.storage_context.persist(persist_dir=str(cfg.index_dir))
    log.info("Indexed %d page-docs in %.1fs", len(docs), time.perf_counter() - start)
    return index


def load_index(cfg: AppConfig) -> VectorStoreIndex:
    """Load the persisted index (no re-embedding: this is what keeps startup fast)."""
    if not (cfg.index_dir / "docstore.json").exists():
        raise FileNotFoundError(
            f"No index at {cfg.index_dir}. Run `ragbot ingest` first."
        )
    configure_models(cfg)
    storage = StorageContext.from_defaults(persist_dir=str(cfg.index_dir))
    return load_index_from_storage(storage)
