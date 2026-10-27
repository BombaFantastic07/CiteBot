"""Shared helpers for evaluation scripts."""
from __future__ import annotations

import json
import statistics
from pathlib import Path


def load_dataset(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        raise SystemExit(
            f"{path} not found. Copy evaluation/dataset.example.jsonl to "
            f"evaluation/dataset.jsonl and fill it with questions about YOUR PDFs."
        )
    rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    if not rows:
        raise SystemExit("Dataset is empty.")
    return rows


def pct_change(before: float, after: float) -> float:
    return (after - before) / before * 100 if before else float("nan")


def percentile(values: list[float], q: float) -> float:
    s = sorted(values)
    return s[min(len(s) - 1, int(round(q * (len(s) - 1))))]


def summarize(values: list[float]) -> dict:
    return {
        "mean": statistics.fmean(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
    }


def save_json(name: str, payload: dict) -> None:
    out = Path(__file__).parent / "results" / name
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved {out}")
