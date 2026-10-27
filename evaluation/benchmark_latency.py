"""End-to-end latency: baseline pipeline vs. optimized pipeline.

Cache is cleared between runs so the comparison measures real retrieval + generation,
not cache hits. Usage: python evaluation/benchmark_latency.py --runs 3
"""
from __future__ import annotations

import argparse

from _common import load_dataset, pct_change, save_json, summarize
from ragbot.config import get_config
from ragbot.engine import RAGEngine
from ragbot.ingest import load_index


def run(engine: RAGEngine, questions: list[str], runs: int) -> list[float]:
    times = []
    engine.ask(questions[0])  # warm-up (connection setup, lazy imports)
    for _ in range(runs):
        for q in questions:
            if engine.cache is not None:
                engine.cache._data.clear()
            times.append(engine.ask(q).latency_ms / 1000)
    return times


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="evaluation/dataset.jsonl")
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args()

    cfg = get_config()
    questions = [r["question"] for r in load_dataset(args.dataset)]
    index = load_index(cfg)

    out = {}
    for mode in ("baseline", "optimized"):
        out[mode] = summarize(run(RAGEngine(cfg, mode=mode, index=index), questions, args.runs))

    print(f"\n{'mode':<12}{'mean(s)':>10}{'p50(s)':>10}{'p95(s)':>10}")
    for mode, s in out.items():
        print(f"{mode:<12}{s['mean']:>10.2f}{s['p50']:>10.2f}{s['p95']:>10.2f}")
    drop = -pct_change(out["baseline"]["mean"], out["optimized"]["mean"])
    print(f"\nMean latency reduction: {drop:.1f}%")
    save_json("latency.json", {**out, "mean_reduction_pct": drop})


if __name__ == "__main__":
    main()
