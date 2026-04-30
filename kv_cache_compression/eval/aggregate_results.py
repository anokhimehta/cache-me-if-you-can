"""
Aggregate per-cell JSONL files in results/kv_compression/ into a single
results/kv_compression_summary.csv with one row per (dataset, method, ratio)
or (dataset, method, image_ratio, text_ratio) cell.

Usage:
    python eval/aggregate_results.py
"""
from __future__ import annotations

import glob
import json
import re
from pathlib import Path

import pandas as pd

RESULTS_DIR = Path("results/kv_compression")
OUT_CSV = Path("results/kv_compression_summary.csv")

UNIFORM_RE = re.compile(r"(?P<dataset>[a-z]+)_(?P<method>[a-z0-9]+)_cr(?P<ratio>\dp\d{2})$")
MODALITY_RE = re.compile(
    r"(?P<dataset>[a-z]+)_modality-(?P<inner>[a-z0-9]+)_i(?P<img>\dp\d{2})_t(?P<txt>\dp\d{2})$"
)


def parse_filename(stem: str) -> dict:
    m = UNIFORM_RE.fullmatch(stem)
    if m:
        return {
            "dataset": m.group("dataset"),
            "method": m.group("method"),
            "compression_ratio": float(m.group("ratio").replace("p", ".")),
            "image_ratio": None,
            "text_ratio": None,
            "inner": None,
        }
    m = MODALITY_RE.fullmatch(stem)
    if m:
        return {
            "dataset": m.group("dataset"),
            "method": "modality",
            "compression_ratio": None,
            "image_ratio": float(m.group("img").replace("p", ".")),
            "text_ratio": float(m.group("txt").replace("p", ".")),
            "inner": m.group("inner"),
        }
    raise ValueError(f"Cannot parse filename stem: {stem}")


def aggregate_cell(path: Path) -> dict:
    samples = [json.loads(line) for line in path.open()]
    n = len(samples)
    if n == 0:
        return {"n_samples": 0}

    def avg(key):
        vals = [s.get(key) for s in samples if s.get(key) is not None]
        return sum(vals) / len(vals) if vals else None

    return {
        "n_samples": n,
        "accuracy": sum(s.get("correct", False) for s in samples) / n,
        "mean_prefill_ms": avg("prefill_ms"),
        "mean_decode_ms": avg("decode_ms"),
        "mean_throughput_tok_per_sec": avg("throughput_tok_per_sec"),
        "peak_gpu_mem_gb": max((s.get("peak_gpu_mem_gb", 0) for s in samples), default=None),
        "mean_image_fraction": avg("image_fraction"),
        "mean_effective_compression": avg("effective_compression_ratio"),
    }


def main():
    files = sorted(RESULTS_DIR.glob("*.jsonl"))
    if not files:
        raise SystemExit(f"No JSONL files found in {RESULTS_DIR}")
    rows = []
    for fp in files:
        meta = parse_filename(fp.stem)
        meta["file"] = fp.name
        meta.update(aggregate_cell(fp))
        rows.append(meta)
    df = pd.DataFrame(rows)
    cols = [
        "dataset", "method", "compression_ratio",
        "image_ratio", "text_ratio", "inner",
        "n_samples", "accuracy",
        "mean_prefill_ms", "mean_decode_ms", "mean_throughput_tok_per_sec",
        "peak_gpu_mem_gb",
        "mean_image_fraction", "mean_effective_compression",
        "file",
    ]
    df = df[cols].sort_values(
        by=["dataset", "method", "compression_ratio", "image_ratio", "text_ratio"]
    ).reset_index(drop=True)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"Wrote {len(df)} rows to {OUT_CSV}")
    print()
    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", 160)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
