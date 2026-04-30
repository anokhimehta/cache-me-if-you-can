"""
Aggregate batched-eval JSONLs into a single CSV.

Each input row already carries batch-level metrics, so this script just
deduplicates per-(dataset, method, cr, batch_size, batch_id) for the timing
columns and joins per-sample accuracy on top.

Output: results/kv_batched_summary.csv
"""

from __future__ import annotations

import glob
import json
import os

import pandas as pd

IN_DIR = "results/kv_batched"
OUT_PATH = "results/kv_batched_summary.csv"


def load_one(path: str) -> pd.DataFrame:
    with open(path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    # Per-batch dedupe for timing
    batch_cols = [
        "dataset", "method", "compression_ratio", "batch_size",
        "batch_id", "batch_total_tokens", "batch_total_wall_ms",
        "batch_ttft_ms", "batch_decode_ms", "batch_throughput_tok_per_sec",
        "peak_gpu_mem_gb", "input_len",
    ]
    batches = df[batch_cols].drop_duplicates(
        subset=["dataset", "method", "compression_ratio", "batch_size", "batch_id"]
    )

    # Per-cell aggregate
    cell = batches.groupby(
        ["dataset", "method", "compression_ratio", "batch_size"], as_index=False
    ).agg(
        n_batches=("batch_id", "count"),
        mean_total_wall_ms=("batch_total_wall_ms", "mean"),
        mean_ttft_ms=("batch_ttft_ms", "mean"),
        mean_decode_ms=("batch_decode_ms", "mean"),
        mean_throughput_tok_per_sec=("batch_throughput_tok_per_sec", "mean"),
        mean_input_len=("input_len", "mean"),
        peak_gpu_mem_gb=("peak_gpu_mem_gb", "max"),  # worst-case across batches
    )

    # Accuracy joined from sample-level rows
    acc = df.groupby(
        ["dataset", "method", "compression_ratio", "batch_size"], as_index=False
    ).agg(
        n_samples=("sample_id", "count"),
        accuracy=("correct", "mean"),
        mean_tokens_per_seq=("num_tokens_generated", "mean"),
    )

    out = cell.merge(
        acc,
        on=["dataset", "method", "compression_ratio", "batch_size"],
        how="left",
    )

    # Throughput per request = total_tokens / wall_time / batch_size? No —
    # throughput here is already tokens-per-second across the batch (production
    # serving metric). We also report per-request latency so users can build
    # both views.
    out["mean_per_request_latency_ms"] = (
        out["mean_total_wall_ms"]
    )  # batch-end-to-end is the latency a client sees

    return out.sort_values(["dataset", "method", "compression_ratio", "batch_size"])


def main():
    files = sorted(glob.glob(os.path.join(IN_DIR, "*.jsonl")))
    if not files:
        print(f"No JSONLs found in {IN_DIR}/")
        return

    parts = []
    for p in files:
        df = load_one(p)
        if not df.empty:
            parts.append(df)
    if not parts:
        print("No non-empty JSONLs.")
        return

    big = pd.concat(parts, ignore_index=True)
    summary = summarize(big)
    summary.to_csv(OUT_PATH, index=False)

    print(f"Wrote {len(summary)} rows to {OUT_PATH}")
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    print(summary.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
