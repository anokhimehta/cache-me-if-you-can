"""
DocVQA Token-Length Bucketing Analysis
=======================================
Parses DocVQA .out log files and groups per-sample results by image token
count. Compares H2O vs Stratified Eviction accuracy within each bucket.

Usage:
    python analysis/bucket_analysis.py \
        --h2o out/docvqa_h2o_cr0.7.out \
        --strat out/docvqa_stratified_cr0.7_vw0.2.out \
        --output results/docvqa_bucket_analysis_cr0.7.txt

The .out files must be DocVQA logs (the only benchmark that logs
"Image tokens : N" per sample).
"""

import re
import argparse
import os
from collections import defaultdict


def parse_docvqa_log(filepath):
    """Parse a DocVQA .out file, return list of (image_tokens, correct)."""
    samples = []
    with open(filepath) as f:
        content = f.read()

    blocks = re.split(r'\n\[\d+\]', content)

    for block in blocks[1:]:  # skip header
        img_match = re.search(r'Image tokens\s*:\s*(\d+)', block)
        correct = '✓' in block
        if img_match:
            samples.append((int(img_match.group(1)), correct))

    return samples


def bucket_results(samples, buckets):
    """Group samples into token count buckets, return accuracy per bucket."""
    results = defaultdict(lambda: {"correct": 0, "total": 0})
    for img_tokens, correct in samples:
        for (lo, hi), label in buckets:
            if lo <= img_tokens < hi:
                results[label]["correct"] += correct
                results[label]["total"] += 1
                break
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h2o",    required=True, help="Path to H2O DocVQA .out file")
    parser.add_argument("--strat",  required=True, help="Path to Stratified DocVQA .out file")
    parser.add_argument("--output", required=True, help="Path to save results .txt")
    args = parser.parse_args()

    buckets = [
        ((0,    1000),  "<1K tokens"),
        ((1000, 2000),  "1K-2K tokens"),
        ((2000, 3000),  "2K-3K tokens"),
        ((3000, 99999), "3K+ tokens"),
    ]

    print(f"Parsing H2O log:        {args.h2o}")
    print(f"Parsing Stratified log: {args.strat}")

    h2o_samples   = parse_docvqa_log(args.h2o)
    strat_samples = parse_docvqa_log(args.strat)

    print(f"H2O samples parsed:        {len(h2o_samples)}")
    print(f"Stratified samples parsed: {len(strat_samples)}")

    h2o_buckets   = bucket_results(h2o_samples,   buckets)
    strat_buckets = bucket_results(strat_samples, buckets)

    lines = []
    lines.append("DocVQA Token-Length Bucketing Analysis")
    lines.append(f"H2O log:        {args.h2o}")
    lines.append(f"Stratified log: {args.strat}")
    lines.append("")
    lines.append(f"{'Bucket':<16} {'H2O Acc':>10} {'Strat Acc':>10} {'Δ':>8} {'N samples':>10}")
    lines.append("-" * 60)

    total_h2o_c = total_strat_c = total_n = 0

    for (lo, hi), label in buckets:
        h = h2o_buckets[label]
        s = strat_buckets[label]
        if h["total"] == 0:
            continue
        h_acc = h["correct"] / h["total"] * 100
        s_acc = s["correct"] / s["total"] * 100
        delta = s_acc - h_acc
        line = (f"{label:<16} {h_acc:>9.1f}% {s_acc:>9.1f}% "
                f"{delta:>+7.2f}pp {h['total']:>10}")
        lines.append(line)
        total_h2o_c   += h["correct"]
        total_strat_c += s["correct"]
        total_n       += h["total"]

    lines.append("-" * 60)
    h_total = total_h2o_c   / total_n * 100
    s_total = total_strat_c / total_n * 100
    lines.append(f"{'Overall':<16} {h_total:>9.1f}% {s_total:>9.1f}% "
                 f"{s_total-h_total:>+7.2f}pp {total_n:>10}")

    output_str = "\n".join(lines)
    print("\n" + output_str)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        f.write(output_str + "\n")
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()