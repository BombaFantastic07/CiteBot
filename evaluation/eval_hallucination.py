"""Hallucination rate via LLM-as-judge faithfulness (LlamaIndex FaithfulnessEvaluator).

A response is "hallucinated" if the judge finds it NOT supported by the retrieved context.
An explicit refusal ("I couldn't find this...") counts as faithful, not hallucinated.
Include some out-of-scope questions in your dataset (no "relevant_file") to test refusals.
Usage: python evaluation/eval_hallucination.py
"""
from __future__ import annotations

import argparse

from _common import load_dataset, pct_change, save_json
from llama_index.core.evaluation import FaithfulnessEvaluator
from ragbot.config import get_config
from ragbot.engine import NO_ANSWER, RAGEngine
from ragbot.ingest import load_index
from ragbot.llm import build_llm


def hallucination_rate(engine: RAGEngine, judge: FaithfulnessEvaluator, questions) -> float:
    bad = 0
    for q in questions:
        ans = engine.ask(q)
        if ans.answer.strip() == NO_ANSWER or not ans.contexts:
            continue
        verdict = judge.evaluate(query=q, response=ans.answer, contexts=ans.contexts)
        if not verdict.passing:
            bad += 1
    return bad / len(questions)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="evaluation/dataset.jsonl")
    args = ap.parse_args()

    cfg = get_config()
    questions = [r["question"] for r in load_dataset(args.dataset)]
    index = load_index(cfg)
    judge = FaithfulnessEvaluator(llm=build_llm(cfg))

    rates = {m: hallucination_rate(RAGEngine(cfg, mode=m, index=index), judge, questions)
             for m in ("baseline", "optimized")}
    drop = -pct_change(rates["baseline"], rates["optimized"])

    print(f"\nQuestions: {len(questions)}")
    for m, r in rates.items():
        print(f"{m:<12} hallucination rate: {r:.1%}")
    print(f"\nRelative reduction: {drop:.1f}%")
    save_json("hallucination.json", {**rates, "relative_reduction_pct": drop})


if __name__ == "__main__":
    main()
