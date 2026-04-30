#!/usr/bin/env python3
import argparse
import datetime
import os
from datasets import load_dataset
from tqdm import tqdm

from qwen_utils import (
    load_model_and_processor,
    generate_answer,
    extract_after_marker,
    normalize_simple_answer,
    numeric_equal,
    extract_first_number,
    save_results,
)


def avg(x):
    return sum(x) / len(x) if x else None


def get_image(ex):
    for key in ["decoded_image", "image"]:
        if key in ex and ex[key] is not None:
            return ex[key]
    for key in ex.keys():
        if key.startswith("image") and ex[key] is not None:
            return ex[key]
    return None


def get_gold_answer(ex):
    for key in ["answer", "final_answer", "gt_answer"]:
        if key in ex and ex[key] is not None:
            return str(ex[key])
    return ""


def is_correct(pred, gold):
    pred_norm = normalize_simple_answer(pred)
    gold_norm = normalize_simple_answer(gold)

    if pred_norm == gold_norm:
        return True

    pred_num = extract_first_number(pred_norm)
    gold_num = extract_first_number(gold_norm)

    if pred_num is not None and gold_num is not None:
        return numeric_equal(pred_num, gold_num)

    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="testmini")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", default="results/")
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--visionzip_keep_ratio", type=float, default=None)
    parser.add_argument("--t2v", action="store_true")
    args = parser.parse_args()

    model, processor = load_model_and_processor()

    ds = load_dataset("AI4Math/MathVista", split=args.split)
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

    processed = 0

    for i, ex in enumerate(tqdm(ds, desc="MathVista")):
        image = get_image(ex)

        if image is None:
            continue

        question = ex.get("question", "")
        prompt = (
            f"Question: {question}\n"
            "Answer concisely. If it is multiple choice, answer only the option letter. "
            "If it is numerical, answer only the final value."
        )


        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        tmp_inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )





        
        num_tokens = tmp_inputs["pixel_values"].shape[0]
        if num_tokens > 2000:
            print(f"Skipping sample {i} (tokens={num_tokens})")
            continue
            
        pred_raw, metrics = generate_answer(
            model,
            processor,
            image,
            prompt,
            max_new_tokens=args.max_new_tokens,
            visionzip_keep_ratio=args.visionzip_keep_ratio,
            
        )

        pred = extract_after_marker(pred_raw).strip()
        gold = get_gold_answer(ex).strip()
        ok = is_correct(pred, gold)

        correct += int(ok)

        if "latency_sec" in metrics:
            latencies.append(metrics["latency_sec"])
        if "prefill_sec" in metrics:
            prefill_times.append(metrics["prefill_sec"])
        if "decode_sec" in metrics:
            decode_times.append(metrics["decode_sec"])
        if metrics.get("peak_memory_gb") is not None:
            peak_memories.append(metrics["peak_memory_gb"])
        if metrics.get("throughput_tok_per_sec") is not None:
            throughputs.append(metrics["throughput_tok_per_sec"])
        if metrics.get("image_tokens_before") is not None:
            image_tokens_before.append(metrics["image_tokens_before"])
        if metrics.get("image_tokens_after") is not None:
            image_tokens_after.append(metrics["image_tokens_after"])

        results.append({
            "index": i,
            "question": question,
            "prediction_raw": pred_raw,
            "prediction_final": pred,
            "gold": gold,
            "correct": ok,
            **metrics,
        })

    total = len(results)
    acc = correct / total if total else 0.0

    payload = {
        "config": {
            "benchmark": "MathVista",
            "split": args.split,
            "limit": args.limit,
            "visionzip_keep_ratio": args.visionzip_keep_ratio,
            
            "max_new_tokens": args.max_new_tokens,
        },
        "summary": {
            "correct": correct,
            "total": total,
            "accuracy": acc,
            "avg_latency_sec": avg(latencies),
            "avg_prefill_sec": avg(prefill_times),
            "avg_decode_sec": avg(decode_times),
            "avg_peak_memory_gb": avg(peak_memories),
            "avg_throughput_tok_per_sec": avg(throughputs),
            "avg_image_tokens_before": avg(image_tokens_before),
            "avg_image_tokens_after": avg(image_tokens_after),
        },
        "results": results,
    }

    summary = payload["summary"]

    print("\n===== SUMMARY METRICS =====")
    
    if summary["accuracy"] is not None:
        print(f"Accuracy                 : {summary['accuracy']:.4f}")
    
    if summary["avg_latency_sec"] is not None:
        print(f"Avg latency (ms)        : {summary['avg_latency_sec'] * 1000:.2f}")
    
    if summary["avg_prefill_sec"] is not None:
        print(f"Avg prefill (ms)        : {summary['avg_prefill_sec'] * 1000:.2f}")
    
    if summary["avg_decode_sec"] is not None:
        print(f"Avg decode (ms)         : {summary['avg_decode_sec'] * 1000:.2f}")
    
    if summary["avg_peak_memory_gb"] is not None:
        print(f"Avg peak memory (GB)    : {summary['avg_peak_memory_gb']:.2f}")
    
    if summary["avg_throughput_tok_per_sec"] is not None:
        print(f"Avg throughput (tok/s)  : {summary['avg_throughput_tok_per_sec']:.2f}")
    
    if summary["avg_image_tokens_before"] is not None:
        print(f"Image tokens BEFORE     : {summary['avg_image_tokens_before']:.1f}")
    
    if summary["avg_image_tokens_after"] is not None:
        print(f"Image tokens AFTER      : {summary['avg_image_tokens_after']:.1f}")
    
    print("============================\n")
    print(f"Accuracy: {correct}/{total} = {acc:.4f}")
    print(f"Avg latency: {payload['summary']['avg_latency_sec']}")
    print(f"Avg peak memory: {payload['summary']['avg_peak_memory_gb']}")
    

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_vz{int(args.visionzip_keep_ratio * 100)}" if args.visionzip_keep_ratio is not None else ""
    
    output_path = os.path.join(args.output, f"mathvista{suffix}_{timestamp}.json")
    save_results(output_path, payload)
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()