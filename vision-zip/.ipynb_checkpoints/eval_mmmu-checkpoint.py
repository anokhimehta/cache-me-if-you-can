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
    correct = 0
    results = []
    latencies = []
    prefill_times = []
    decode_times = []
    peak_memories = []
    throughputs = []
    image_tokens_before = []
    image_tokens_after = []
    MMMU_SUBJECTS = [
    "Accounting",
    "Agriculture",
    "Architecture_and_Engineering",
    "Art",
    "Art_Theory",
    "Basic_Medical_Science",
    "Biology",
    "Chemistry",
    "Clinical_Medicine",
    "Computer_Science",
    "Design",
    "Diagnostics_and_Laboratory_Medicine",
    "Economics",
    "Electronics",
    "Energy_and_Power",
    "Finance",
    "Geography",
    "History",
    "Literature",
    "Manage",
    "Marketing",
    "Materials",
    "Math",
    "Mechanical_Engineering",
    "Music",
    "Pharmacy",
    "Physics",
    "Psychology",
    "Public_Health",
    "Sociology",
    ]
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", default="results/")
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--visionzip_keep_ratio", type=float, default=None)
    parser.add_argument("--subjects", nargs="+", default=["Accounting"])
    args = parser.parse_args()

    model, processor = load_model_and_processor()
    
    subjects = MMMU_SUBJECTS if args.subjects == ["all"] else args.subjects
    subject_stats = {}
    
    for subject in subjects:
        print(f"\nRunning MMMU subject: {subject}")
        subject_stats[subject] = {
        "correct": 0,
        "total": 0,
        "latencies": [],
        "prefill_times": [],
        "decode_times": [],
        "peak_memories": [],
        "throughputs": [],
        "image_tokens_before": [],
        "image_tokens_after": [],
        }
        
        ds = load_dataset("MMMU/MMMU", subject, split=args.split)
        if args.limit is not None:
            ds = ds.select(range(min(args.limit, len(ds))))
  
        for i, ex in enumerate(tqdm(ds, desc=f"MMMU-{subject}")):
    
            #-----image extraction------#
            image = None
            for k in ex.keys():
                if k.startswith("image"):
                    image = ex[k]
                    break
            if image is None:
                continue
    
            #------prompt-------#
            options = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(ex["options"])])
            prompt = (
                f"Question: {ex['question']}\n"
                f"{options}\n"
                "Answer with only the correct option (A/B/C/D)."
            )
            pred_raw, metrics = generate_answer(
                model, processor, image, prompt,
                max_new_tokens=args.max_new_tokens,
                visionzip_keep_ratio=args.visionzip_keep_ratio,
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
    
            #---eval----#
            pred = extract_after_marker(pred_raw).strip().upper()
            gold = ex["answer"].strip().upper()
    
            ok = pred == gold
            correct += int(ok)
            subject_stats[subject]["correct"] += int(ok)
            subject_stats[subject]["total"] += 1
    
    
            if "latency_sec" in metrics:
              subject_stats[subject]["latencies"].append(metrics["latency_sec"])
    
            if "prefill_sec" in metrics:
              subject_stats[subject]["prefill_times"].append(metrics["prefill_sec"])
    
            if "decode_sec" in metrics:
              subject_stats[subject]["decode_times"].append(metrics["decode_sec"])
    
            if metrics.get("peak_memory_gb") is not None:
              subject_stats[subject]["peak_memories"].append(metrics["peak_memory_gb"])
    
            if metrics.get("generated_tokens") and metrics.get("latency_sec"):
                subject_stats[subject]["throughputs"].append(
                    metrics["generated_tokens"] / metrics["latency_sec"]
                )
    
            if metrics.get("image_tokens_before") is not None:
                subject_stats[subject]["image_tokens_before"].append(metrics["image_tokens_before"])
    
            if metrics.get("image_tokens_after") is not None:
                subject_stats[subject]["image_tokens_after"].append(metrics["image_tokens_after"])
    
            
    
            results.append({
                "subject":subject,
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

    per_subject_summary = {}

    for subject, s in subject_stats.items():
        total_s = s["total"]
        correct_s = s["correct"]

        per_subject_summary[subject] = {
        "correct": correct_s,
        "total": total_s,
        "accuracy": correct_s / total_s if total_s else 0.0,
        "avg_latency_sec": avg(s["latencies"]),
        "avg_prefill_sec": avg(s["prefill_times"]),
        "avg_decode_sec": avg(s["decode_times"]),
        "avg_peak_memory_gb": avg(s["peak_memories"]),
        "avg_throughput_tok_per_sec": avg(s["throughputs"]),
        "avg_image_tokens_before": avg(s["image_tokens_before"]),
        "avg_image_tokens_after": avg(s["image_tokens_after"]),
    }





    

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

    if avg_image_tokens_before is not None:
        print(f"Avg image tokens after: {avg_image_tokens_after:.1f}")

    payload = {
        "config": {
        "visionzip_keep_ratio": args.visionzip_keep_ratio,
        "max_new_tokens": args.max_new_tokens,
        "split": args.split,
        "limit": args.limit,
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
    "per_subject_summary": per_subject_summary,
    "results": results,
    }
    
    print(f"Accuracy: {correct}/{total} = {acc:.4f}")


    print("\nPer-subject results:")
    for subject, s in per_subject_summary.items():
        print(
            f"{subject:35s} "
            f"Acc={s['accuracy']:.4f} "
            f"{s['correct']}/{s['total']}"
        )





    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    suffix = ""
    if args.visionzip_keep_ratio is not None:
        suffix = f"_vz{int(args.visionzip_keep_ratio * 100)}"
    filename = f"mmmu_dropping_{suffix}_{timestamp}.json"
    os.makedirs(args.output, exist_ok=True)
    output_path = os.path.join(args.output, filename)
    save_results(output_path,payload)
   
    print(f"Saved to: {output_path}")

if __name__ == "__main__":
    main()