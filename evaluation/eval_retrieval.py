"""Retrieval quality: dense vector search vs. BM25 keyword baseline.

Metrics: Hit@k (is a relevant file in the top-k?) and MRR.
Dataset rows: {"question": "...", "relevant_file": "name.pdf"}
Usage: python evaluation/eval_retrieval.py --dataset evaluation/dataset.jsonl --k 4
"""
from __future__ import annotations

import argparse

from _common import load_dataset, pct_change, save_json
from llama_index.retrievers.bm25 import BM25Retriever
from ragbot.config import get_config
from ragbot.ingest import load_index


def score(retriever, rows, k):
    hits, rr = 0, 0.0
    for row in rows:
        nodes = retriever.retrieve(row["question"])[:k]
        files = [n.node.metadata.get("file_name") for n in nodes]
        if row["relevant_file"] in files:
            hits += 1
            rr += 1.0 / (files.index(row["relevant_file"]) + 1)
    n = len(rows)
    return {"hit_rate": hits / n, "mrr": rr / n}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="evaluation/dataset.jsonl")
    ap.add_argument("--k", type=int, default=4)
    args = ap.parse_args()

    cfg = get_config()
    rows = load_dataset(args.dataset)
    index = load_index(cfg)

    dense = index.as_retriever(similarity_top_k=args.k)
    bm25 = BM25Retriever.from_defaults(docstore=index.docstore, similarity_top_k=args.k)

    res = {"keyword_bm25": score(bm25, rows, args.k), "vector": score(dense, rows, args.k)}
    gain_hit = pct_change(res["keyword_bm25"]["hit_rate"], res["vector"]["hit_rate"])
    gain_mrr = pct_change(res["keyword_bm25"]["mrr"], res["vector"]["mrr"])

    print(f"\nQuestions: {len(rows)}   k={args.k}")
    print(f"{'method':<16}{'Hit@k':>10}{'MRR':>10}")
    for name, m in res.items():
        print(f"{name:<16}{m['hit_rate']:>10.3f}{m['mrr']:>10.3f}")
    print(f"\nVector vs keyword: Hit@k {gain_hit:+.1f}%   MRR {gain_mrr:+.1f}%")
    save_json("retrieval.json", {"k": args.k, "n": len(rows), **res,
                                 "hit_rate_gain_pct": gain_hit, "mrr_gain_pct": gain_mrr})


if __name__ == "__main__":
    main()
