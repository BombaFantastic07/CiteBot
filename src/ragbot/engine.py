"""Query engine with two modes.

optimized (default):
    top-k vector retrieval -> similarity cutoff -> strict grounded prompt
    -> compact synthesis (fewest LLM calls) -> LRU cache -> source citations.
baseline:
    larger top-k, no cutoff, default prompt, iterative "refine" synthesis, no cache.
    Used by the evaluation scripts for honest before/after comparisons.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterator, List, Optional

from llama_index.core import PromptTemplate, VectorStoreIndex
from llama_index.core.response_synthesizers import ResponseMode, get_response_synthesizer

from .cache import LRUCache
from .config import AppConfig
from .ingest import load_index

NO_ANSWER = "I couldn't find this in the provided documents."

GROUNDED_PROMPT = PromptTemplate(
    "You are a careful assistant that answers strictly from the provided context.\n"
    "Rules:\n"
    "1. Use ONLY the context below. Do not use outside knowledge.\n"
    "2. If the context does not contain the answer, reply exactly: " + NO_ANSWER + "\n"
    "3. Be concise. Cite sources inline as [file, p.N].\n\n"
    "Context:\n{context_str}\n\n"
    "Question: {query_str}\n"
    "Answer:"
)


@dataclass
class Answer:
    question: str
    answer: str
    sources: List[dict] = field(default_factory=list)
    contexts: List[str] = field(default_factory=list)
    latency_ms: float = 0.0
    cached: bool = False
    grounded: bool = True  # False when we refused due to weak retrieval

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "sources": self.sources,
            "latency_ms": round(self.latency_ms, 1),
            "cached": self.cached,
            "grounded": self.grounded,
        }


class RAGEngine:
    def __init__(
        self,
        cfg: AppConfig,
        mode: str = "optimized",
        index: Optional[VectorStoreIndex] = None,
    ) -> None:
        if mode not in {"optimized", "baseline"}:
            raise ValueError("mode must be 'optimized' or 'baseline'")
        self.cfg = cfg
        self.mode = mode
        self.index = index or load_index(cfg)

        optimized = mode == "optimized"
        self.top_k = cfg.top_k if optimized else max(cfg.top_k * 2, 8)
        self.cutoff = cfg.similarity_cutoff if optimized else 0.0
        self.cache: Optional[LRUCache[Answer]] = LRUCache(cfg.cache_size) if optimized else None

        self.retriever = self.index.as_retriever(similarity_top_k=self.top_k)
        if optimized:
            self.synth = get_response_synthesizer(
                response_mode=ResponseMode.COMPACT, text_qa_template=GROUNDED_PROMPT
            )
            self.stream_synth = get_response_synthesizer(
                response_mode=ResponseMode.COMPACT,
                text_qa_template=GROUNDED_PROMPT,
                streaming=True,
            )
        else:
            self.synth = get_response_synthesizer(response_mode=ResponseMode.REFINE)
            self.stream_synth = None

    # ------------------------------------------------------------------ retrieval
    def retrieve(self, question: str):
        nodes = self.retriever.retrieve(question)
        if self.cutoff > 0:
            nodes = [n for n in nodes if n.score is not None and n.score >= self.cutoff]
        return nodes

    @staticmethod
    def _sources(nodes) -> List[dict]:
        seen, out = set(), []
        for n in nodes:
            md = n.node.metadata
            key = (md.get("file_name"), md.get("page_label"))
            if key in seen:
                continue
            seen.add(key)
            out.append(
                {
                    "file": md.get("file_name"),
                    "page": md.get("page_label"),
                    "score": round(float(n.score or 0.0), 3),
                }
            )
        return out

    # ------------------------------------------------------------------ answering
    def ask(self, question: str) -> Answer:
        start = time.perf_counter()

        if self.cache is not None:
            hit = self.cache.get(question)
            if hit is not None:
                return Answer(
                    **{**hit.__dict__, "cached": True,
                       "latency_ms": (time.perf_counter() - start) * 1000}
                )

        nodes = self.retrieve(question)
        if not nodes:  # nothing relevant -> refuse instead of letting the LLM guess
            result = Answer(question, NO_ANSWER, grounded=False)
        else:
            response = self.synth.synthesize(question, nodes)
            result = Answer(
                question=question,
                answer=str(response).strip(),
                sources=self._sources(nodes),
                contexts=[n.node.get_content() for n in nodes],
            )
        result.latency_ms = (time.perf_counter() - start) * 1000

        if self.cache is not None:
            self.cache.put(question, result)
        return result

    def stream(self, question: str) -> Iterator[str]:
        """Token streaming: cuts perceived latency to time-to-first-token."""
        if self.stream_synth is None:
            yield self.ask(question).answer
            return
        nodes = self.retrieve(question)
        if not nodes:
            yield NO_ANSWER
            return
        yield from self.stream_synth.synthesize(question, nodes).response_gen
