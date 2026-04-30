"""
Batched KV-cache compression evaluator for Qwen3-VL-4B.

This is the production-style sibling of eval_kv_methods.py. Instead of running
samples one at a time it processes them in batches of size N, then reports
*throughput* (tokens generated per wall-second per GPU) in addition to the
per-sample latency metrics. This is the relevant metric for serving and is
what tests the Finding 2 hypothesis from the project README ("does H2O's
attention-stat overhead get amortized across a batch?").

Outputs one JSONL row per sample (so the existing accuracy heuristics work
unchanged) plus batch-level fields that the matched aggregator joins on.

Usage
-----
    # h2o at cr=0.5, batch=8, on mmmu, 32 samples
    python -m eval.eval_kv_batched \\
        --method h2o --dataset mmmu --compression_ratio 0.5 \\
        --batch_size 8 --max_samples 32

    # streaming baseline at cr=0 (the absolute-fastest reference)
    python -m eval.eval_kv_batched \\
        --method streaming --dataset mmmu --compression_ratio 0.0 \\
        --batch_size 8 --max_samples 32

Output
------
    results/kv_batched/<dataset>_<method>_cr<ratio>_bs<batch>.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch
from datasets import load_dataset

from src.load_model import load_model_and_processor
from src.utils import reset_gpu_memory, get_peak_gpu_memory_mb
from src.kv_compression import make_press

# Reuse the dataset/sample/scoring logic from the unified single-sample evaluator
# so the batched run is comparable to the existing 84-cell numbers.
from eval.eval_kv_methods import (
    DATASET_CONFIGS,
    extract_sample,
    is_correct,
    load_benchmark,
)


# ---------------------------------------------------------------------------
# Batched input building
# ---------------------------------------------------------------------------

def build_batched_inputs(processor, model, samples_data):
    """
    samples_data : list[(image, question)]
    Returns a HuggingFace BatchEncoding ready for model.generate(...).
    """
    texts, images = [], []
    for image, question in samples_data:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            }
        ]
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        texts.append(text)
        images.append(image)

    # Some processors expect images as a flat list, some as nested list-of-lists.
    # Qwen3-VL accepts the flat form alongside one text per sample.
    inputs = processor(
        text=texts,
        images=images,
        padding=True,
        return_tensors="pt",
    ).to(model.device)
    return inputs


def count_new_tokens(new_tokens: torch.Tensor, pad_id: int, eos_id: int) -> list[int]:
    """For each row of new_tokens, count tokens up to (but not including) the
    first pad or eos. Used to compute per-sequence emitted token counts."""
    counts = []
    for row in new_tokens.tolist():
        n = 0
        for tok in row:
            if tok == pad_id or tok == eos_id:
                break
            n += 1
        counts.append(n)
    return counts


# ---------------------------------------------------------------------------
# Single-batch run
# ---------------------------------------------------------------------------

def run_batch(model, processor, samples_data, max_new_tokens, press):
    """Run one batched generate() under `press` and return predictions + timing.

    Returns a dict with:
      predictions          : list[str]
      tokens_per_seq       : list[int]
      total_tokens         : int
      ttft_ms              : float   (time-to-first-token across the batch)
      total_wall_ms        : float   (prefill + decode end-to-end)
      decode_ms            : float   (total - ttft)
      throughput_tok_per_s : float   (total_tokens / total_wall)
      peak_gpu_mem_gb      : float
      input_len            : int
    """
    inputs = build_batched_inputs(processor, model, samples_data)

    pad_id = processor.tokenizer.pad_token_id
    eos_id = processor.tokenizer.eos_token_id

    # Time-to-first-token via a LogitsProcessor that fires on the first decode step.
    from transformers import LogitsProcessor

    class TTFTHook(LogitsProcessor):
        def __init__(self):
            self.t = None
            self.input_len = inputs["input_ids"].shape[1]

        def __call__(self, input_ids, scores):
            if self.t is None and input_ids.shape[1] > self.input_len:
                torch.cuda.synchronize()
                self.t = time.perf_counter()
            return scores

    ttft_hook = TTFTHook()

    reset_gpu_memory()
    torch.cuda.synchronize()
    t_start = time.perf_counter()

    with press(model):
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            logits_processor=[ttft_hook],
            pad_token_id=pad_id if pad_id is not None else eos_id,
        )

    torch.cuda.synchronize()
    t_end = time.perf_counter()

    input_len = inputs["input_ids"].shape[1]
    new_tokens = outputs[:, input_len:]
    tokens_per_seq = count_new_tokens(new_tokens, pad_id or -1, eos_id or -1)
    total_tokens = sum(tokens_per_seq)

    predictions = processor.tokenizer.batch_decode(
        new_tokens, skip_special_tokens=True
    )

    total_wall_ms = (t_end - t_start) * 1000
    ttft_ms = (
        (ttft_hook.t - t_start) * 1000 if ttft_hook.t is not None else total_wall_ms
    )
    decode_ms = max(total_wall_ms - ttft_ms, 0.0)
    wall_sec = max(t_end - t_start, 1e-9)
    throughput = total_tokens / wall_sec
    peak_mem_gb = get_peak_gpu_memory_mb() / 1024.0

    return {
        "predictions": predictions,
        "tokens_per_seq": tokens_per_seq,
        "total_tokens": total_tokens,
        "ttft_ms": ttft_ms,
        "total_wall_ms": total_wall_ms,
        "decode_ms": decode_ms,
        "throughput_tok_per_sec": throughput,
        "peak_gpu_mem_gb": peak_mem_gb,
        "input_len": input_len,
    }


# ---------------------------------------------------------------------------
# Press builder (subset of eval_kv_methods — modality-aware not supported in
# batched mode yet because the modality mask is built per-sample)
# ---------------------------------------------------------------------------

def build_press(args):
    if args.method == "h2o":
        return make_press("h2o", compression_ratio=args.compression_ratio)
    if args.method == "streaming":
        return make_press(
            "streaming",
            compression_ratio=args.compression_ratio,
            n_sink=args.n_sink,
        )
    if args.method == "snapkv":
        return make_press(
            "snapkv",
            compression_ratio=args.compression_ratio,
            window_size=args.window_size,
            kernel_size=args.kernel_size,
        )
    if args.method == "pyramid":
        return make_press(
            "pyramid",
            compression_ratio=args.compression_ratio,
            window_size=args.window_size,
            kernel_size=args.kernel_size,
            beta=args.beta,
        )
    raise ValueError(f"Method {args.method!r} not supported in batched mode yet.")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Batched KV-cache compression evaluator.")
    p.add_argument("--method", required=True,
                   choices=["h2o", "streaming", "snapkv", "pyramid"])
    p.add_argument("--dataset", required=True, choices=list(DATASET_CONFIGS.keys()))
    p.add_argument("--compression_ratio", type=float, default=0.5)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--max_samples", type=int, default=32)

    # Method-specific knobs (kept identical to eval_kv_methods.py)
    p.add_argument("--n_sink", type=int, default=4)
    p.add_argument("--window_size", type=int, default=64)
    p.add_argument("--kernel_size", type=int, default=5)
    p.add_argument("--beta", type=int, default=20)
    return p.parse_args()


def results_path(args) -> str:
    out_dir = "results/kv_batched"
    os.makedirs(out_dir, exist_ok=True)
    r = f"{args.compression_ratio:.2f}".replace(".", "p")
    return f"{out_dir}/{args.dataset}_{args.method}_cr{r}_bs{args.batch_size}.jsonl"


def main():
    args = parse_args()
    out_path = results_path(args)

    print("=" * 80)
    print("Batched KV Compression Eval")
    print("=" * 80)
    print(f"  Method       : {args.method}")
    print(f"  Dataset      : {args.dataset}")
    print(f"  Compression  : {args.compression_ratio}")
    print(f"  Batch size   : {args.batch_size}")
    print(f"  Max samples  : {args.max_samples}")
    print(f"  Output       : {out_path}")
    print()

    print("Loading model and processor...")
    model, processor = load_model_and_processor()
    press = build_press(args)

    print(f"Loading dataset: {args.dataset} ...")
    dataset = load_benchmark(args.dataset)
    cfg = DATASET_CONFIGS[args.dataset]

    n_eval = min(args.max_samples, len(dataset))
    n_correct, n_total = 0, 0

    with open(out_path, "w") as fout:
        for batch_start in range(0, n_eval, args.batch_size):
            batch_end = min(batch_start + args.batch_size, n_eval)
            batch_indices = list(range(batch_start, batch_end))

            samples = [dataset[i] for i in batch_indices]
            samples_data = [extract_sample(args.dataset, s) for s in samples]
            iq = [(d[0], d[1]) for d in samples_data]

            t_pre = time.perf_counter()
            try:
                res = run_batch(model, processor, iq, cfg["max_new_tokens"], press)
            except torch.cuda.OutOfMemoryError as e:
                print(f"[OOM] batch starting at {batch_start} (size {len(iq)}): {e}")
                torch.cuda.empty_cache()
                continue
            t_post = time.perf_counter()

            for i, (image, question, gt) in enumerate(samples_data):
                pred = res["predictions"][i]
                correct = is_correct(pred, gt)
                if correct:
                    n_correct += 1
                n_total += 1

                row = {
                    "sample_id": batch_indices[i],
                    "batch_id": batch_start // args.batch_size,
                    "batch_size": len(iq),
                    "position_in_batch": i,
                    "dataset": args.dataset,
                    "method": args.method,
                    "compression_ratio": args.compression_ratio,
                    "question": question,
                    "ground_truth": gt,
                    "prediction": pred,
                    "num_tokens_generated": res["tokens_per_seq"][i],
                    # batch-level metrics (same value for every row in the batch)
                    "batch_total_tokens": res["total_tokens"],
                    "batch_total_wall_ms": round(res["total_wall_ms"], 2),
                    "batch_ttft_ms": round(res["ttft_ms"], 2),
                    "batch_decode_ms": round(res["decode_ms"], 2),
                    "batch_throughput_tok_per_sec": round(res["throughput_tok_per_sec"], 2),
                    "peak_gpu_mem_gb": round(res["peak_gpu_mem_gb"], 3),
                    "input_len": res["input_len"],
                    "correct": correct,
                }
                fout.write(json.dumps(row) + "\n")
                fout.flush()

            print(
                f"[batch {batch_start // args.batch_size:>3}] "
                f"size={len(iq)}  "
                f"ttft={res['ttft_ms']:>7.1f}ms  "
                f"wall={res['total_wall_ms']:>7.1f}ms  "
                f"throughput={res['throughput_tok_per_sec']:>6.1f} tok/s  "
                f"peakmem={res['peak_gpu_mem_gb']:.2f} GB  "
                f"loop={(t_post - t_pre)*1000:>7.1f}ms"
            )

    accuracy = n_correct / n_total if n_total > 0 else 0.0
    print("=" * 80)
    print(f"FINAL ACCURACY : {n_correct}/{n_total} = {accuracy:.1%}")
    print(f"Results saved to {out_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
