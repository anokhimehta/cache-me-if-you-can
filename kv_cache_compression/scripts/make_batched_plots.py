"""
Plot Stage-1 production benchmarks: throughput vs batch size, and the
throughput–latency Pareto frontier per dataset.

Inputs : results/kv_batched_summary.csv
Outputs: figures/fig5_throughput_vs_batch.{png,pdf}
         figures/fig6_pareto_throughput_latency.{png,pdf}
         figures/fig7_ttft_vs_batch.{png,pdf}
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

CSV = "results/kv_batched_summary.csv"
OUT = "figures"
os.makedirs(OUT, exist_ok=True)


def style_for(method: str, cr: float):
    base_color = {"h2o": "tab:blue", "streaming": "tab:red",
                  "snapkv": "tab:green", "pyramid": "tab:orange"}.get(method, "k")
    linestyle = "-" if cr == 0.0 else "--"
    return base_color, linestyle


def fig5_throughput_vs_batch(df: pd.DataFrame):
    datasets = sorted(df.dataset.unique())
    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 4),
                             sharey=False)
    if len(datasets) == 1:
        axes = [axes]
    for ax, ds in zip(axes, datasets):
        sub = df[df.dataset == ds]
        for (method, cr), g in sub.groupby(["method", "compression_ratio"]):
            g = g.sort_values("batch_size")
            color, ls = style_for(method, cr)
            ax.plot(g.batch_size, g.mean_throughput_tok_per_sec,
                    marker="o", linestyle=ls, color=color,
                    label=f"{method} cr={cr:.1f}")
        ax.set_title(ds)
        ax.set_xlabel("Batch size")
        ax.set_ylabel("Throughput (tok/s)")
        ax.set_xscale("log", base=2)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Throughput vs batch size")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig5_throughput_vs_batch.png", dpi=150, bbox_inches="tight")
    fig.savefig(f"{OUT}/fig5_throughput_vs_batch.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved fig5")


def fig6_pareto(df: pd.DataFrame):
    datasets = sorted(df.dataset.unique())
    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 4),
                             sharey=False)
    if len(datasets) == 1:
        axes = [axes]
    for ax, ds in zip(axes, datasets):
        sub = df[df.dataset == ds]
        for (method, cr), g in sub.groupby(["method", "compression_ratio"]):
            color, ls = style_for(method, cr)
            ax.plot(g.mean_per_request_latency_ms,
                    g.mean_throughput_tok_per_sec,
                    marker="o", linestyle=ls, color=color,
                    label=f"{method} cr={cr:.1f}")
            for _, row in g.iterrows():
                ax.annotate(f"bs={int(row.batch_size)}",
                            (row.mean_per_request_latency_ms,
                             row.mean_throughput_tok_per_sec),
                            fontsize=6, alpha=0.7,
                            xytext=(3, 3), textcoords="offset points")
        ax.set_title(ds)
        ax.set_xlabel("End-to-end batch latency (ms)")
        ax.set_ylabel("Throughput (tok/s)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Throughput vs latency Pareto (label = batch size)")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig6_pareto_throughput_latency.png", dpi=150, bbox_inches="tight")
    fig.savefig(f"{OUT}/fig6_pareto_throughput_latency.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved fig6")


def fig7_ttft(df: pd.DataFrame):
    datasets = sorted(df.dataset.unique())
    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 4),
                             sharey=False)
    if len(datasets) == 1:
        axes = [axes]
    for ax, ds in zip(axes, datasets):
        sub = df[df.dataset == ds]
        for (method, cr), g in sub.groupby(["method", "compression_ratio"]):
            g = g.sort_values("batch_size")
            color, ls = style_for(method, cr)
            ax.plot(g.batch_size, g.mean_ttft_ms,
                    marker="o", linestyle=ls, color=color,
                    label=f"{method} cr={cr:.1f}")
        ax.set_title(ds)
        ax.set_xlabel("Batch size")
        ax.set_ylabel("Mean TTFT (ms)")
        ax.set_xscale("log", base=2)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Time-to-first-token vs batch size")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig7_ttft_vs_batch.png", dpi=150, bbox_inches="tight")
    fig.savefig(f"{OUT}/fig7_ttft_vs_batch.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved fig7")


def main():
    if not os.path.exists(CSV):
        raise SystemExit(f"Missing {CSV}. Run aggregate_batched.py first.")
    df = pd.read_csv(CSV)
    if df.empty:
        raise SystemExit("Summary CSV is empty.")
    fig5_throughput_vs_batch(df)
    fig6_pareto(df)
    fig7_ttft(df)
    print(f"All figures written to {OUT}/")


if __name__ == "__main__":
    main()
