#!/usr/bin/env python3
import argparse
import os
from datasets import load_dataset
from tqdm import tqdm
import datetime

from qwen_utils import (
    load_model_and_processor,
    generate_answer,
    normalize_simple_answer,
    extract_after_marker,
    save_results,
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", default="results/")
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--visionzip_keep_ratio", type=float, default=None)
    parser.add_argument("--t2v", action="store_true")
    parser.add_argument("--t2v_layer1_ratio", type=float, default=0.5)
    parser.add_argument("--t2v_layer2_ratio", type=float, default=0.25)
    
    args = parser.parse_args()

    model, processor = load_model_and_processor()
    ds = load_dataset("xai-org/RealworldQA", split=args.split)

    if args.limit is not None:
        ds = ds.select(range(min(args.limit, len(ds))))

    correct = 0
    results = []
    latencies = []
    prefill_times = []
    decode_times = []
    peak_memories = []
    throughputs = []
    image_tokens_before = []
    image_tokens_after = []

    for i, ex in enumerate(tqdm(ds, desc="RealWorldQA")):
        prompt = (f"Question: {ex['question']}\n"
            "Answer with only the final answer. No explanation.")


        t2v_keep_ratios = None
        if args.t2v:
            t2v_keep_ratios = {
                14: args.t2v_layer1_ratio,
                21: args.t2v_layer2_ratio,
            }


        
        pred_raw, metrics = generate_answer(
            model, processor, ex["image"], prompt,
            max_new_tokens=args.max_new_tokens,
            visionzip_keep_ratio=args.visionzip_keep_ratio,
            t2v_keep_ratios=t2v_keep_ratios,
        )

        # collect metrics
        if "latency_sec" in metrics:
            latencies.append(metrics["latency_sec"])
        if "prefill_sec" in metrics:
            prefill_times.append(metrics["prefill_sec"])
        
        if "decode_sec" in metrics:
            decode_times.append(metrics["decode_sec"])
        
        if metrics.get("peak_memory_gb") is not None:
            peak_memories.append(metrics["peak_memory_gb"])
        
        if metrics.get("generated_tokens") is not None and metrics.get("latency_sec") > 0:
            throughputs.append(metrics["generated_tokens"] / metrics["latency_sec"])
        
        if metrics.get("image_tokens_before") is not None:
            image_tokens_before.append(metrics["image_tokens_before"])

        if metrics.get("image_tokens_after") is not None:
            image_tokens_after.append(metrics["image_tokens_after"])

        pred = extract_after_marker(pred_raw)
        gold = ex["answer"]

        ok = normalize_simple_answer(pred) == normalize_simple_answer(gold)
        correct += int(ok)

        results.append({
            "index": i,
            "question": ex["question"],
            "prediction_raw": pred_raw,
            "prediction_final": pred,
            "gold": gold,
            "correct": ok,
            **metrics,
        })

    total = len(results)
    acc = correct / total if total else 0.0


    def avg(x):
        return sum(x) / len(x) if x else None

    avg_prefill = avg(prefill_times)
    avg_decode = avg(decode_times)
    avg_peak_memory = avg(peak_memories)
    avg_throughput = avg(throughputs)
    # avg_image_tokens = avg(image_token_counts)
    avg_image_tokens_before = avg(image_tokens_before)
    avg_image_tokens_after = avg(image_tokens_after)
    avg_latency = avg(latencies)

    if avg_latency is not None:
        print(f"Avg latency     : {avg_latency * 1000:.1f} ms")

    if avg_prefill is not None:
        print(f"Avg prefill     : {avg_prefill * 1000:.1f} ms")

    if avg_decode is not None:
        print(f"Avg decode      : {avg_decode * 1000:.1f} ms")
    
    if avg_peak_memory is not None:
        print(f"Avg peak memory : {avg_peak_memory:.2f} GB")
    
    if avg_throughput is not None:
        print(f"Avg throughput  : {avg_throughput:.1f} tok/s")
    
    if avg_image_tokens_before is not None:
        print(f"Avg image tokens before: {avg_image_tokens_before:.1f}")

    if avg_image_tokens_after is not None:
        print(f"Avg image tokens after: {avg_image_tokens_after:.1f}")

    payload = {
        "config": {
        "visionzip_keep_ratio": args.visionzip_keep_ratio,
        "max_new_tokens": args.max_new_tokens,
        "split": args.split,
        "limit": args.limit,
        "t2v": args.t2v,
        "t2v_keep_ratios": t2v_keep_ratios,
    },
        
    "summary": {
        "correct": correct,
        "total": total,
        "accuracy": acc,
        "avg_prefill_sec": avg_prefill,
        "avg_decode_sec": avg_decode,
        "avg_peak_memory_gb": avg_peak_memory,
        "avg_throughput_tok_per_sec": avg_throughput,
        "avg_image_tokens_before": avg_image_tokens_before,
        "avg_image_tokens_after": avg_image_tokens_after,
        "avg_latency_sec": avg_latency,
        
    },
    "results": results,
    }
    
    print(f"Accuracy: {correct}/{total} = {acc:.4f}")
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    suffix = ""
    if args.visionzip_keep_ratio is not None:
        suffix = f"_vz{int(args.visionzip_keep_ratio * 100)}"
    filename = f"realworldqa_duetVLM_{suffix}_{timestamp}.json"
    os.makedirs(args.output, exist_ok=True)
    output_path = os.path.join(args.output, filename)
    save_results(output_path,payload)
   
    print(f"Saved to: {output_path}")

if __name__ == "__main__":
    main()