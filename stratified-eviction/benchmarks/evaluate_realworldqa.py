import argparse
import importlib
import json
import os
import re

from datasets import load_dataset
from model import load_model, run_inference


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--method", default=None, help="methods module name, e.g. h2o or stratified_eviction")
    p.add_argument("--compression_ratio", type=float, default=0.5)
    p.add_argument("--vision_weight", type=float, default=0.2)
    return p.parse_args()


def extract_letter(text):
    match = re.search(r'\b([A-E])\b', text.upper())
    return match.group(1) if match else text[0].upper() if text else ""


def is_correct(prediction, correct):
    if len(correct) == 1:
        return extract_letter(prediction) == correct.upper()
    return correct.lower() in prediction.lower()


def main():
    args = parse_args()

    if args.method is not None:
        mod = importlib.import_module(f"methods.{args.method}")
        press = mod.get_press(compression_ratio=args.compression_ratio, vision_weight=args.vision_weight)
    else:
        press = None

    model, processor = load_model()
    ds = load_dataset("xai-org/RealworldQA", split="test")

    correct_count    = 0
    total            = 0
    total_prefill    = 0
    total_decode     = 0
    total_mem        = 0
    total_throughput = 0
    total_img_tokens = 0

    for sample in ds:
        image    = sample["image"]
        question = sample["question"]
        correct  = sample["answer"]

        prediction, metrics = run_inference(model, processor, image, question, max_new_tokens=16, press=press)
        predicted_letter = extract_letter(prediction)

        total_prefill    += metrics["prefill_ms"]
        total_decode     += metrics["decode_ms"]
        total_mem        += metrics["peak_mem_gb"]
        total_throughput += metrics["throughput_tokps"]
        total_img_tokens += metrics["num_image_tokens"]

        correct_flag = is_correct(prediction, correct)
        correct_count += correct_flag
        total += 1

        print(f"[{total}] Predicted: {predicted_letter} | Correct: {correct} | {'✓' if correct_flag else '✗'}")

    accuracy = correct_count / total * 100
    print(f"\nAccuracy          : {correct_count}/{total} = {accuracy:.1f}%")
    print(f"Avg prefill       : {total_prefill/total:.1f} ms")
    print(f"Avg decode        : {total_decode/total:.1f} ms")
    print(f"Avg peak memory   : {total_mem/total:.2f} GB")
    print(f"Avg throughput    : {total_throughput/total:.1f} tok/s")
    print(f"Avg image tokens  : {total_img_tokens/total:.1f}")

    method_tag = args.method or "baseline"
    tag = f"{method_tag}_cr{args.compression_ratio}"
    if args.method == "stratified_eviction":
        tag += f"_vw{args.vision_weight}"
    if args.method is None:
        tag = "baseline"

    os.makedirs("results", exist_ok=True)
    out_path = os.path.join("results", f"realworldqa_{tag}.json")
    with open(out_path, "w") as f:
        json.dump({
            "benchmark":         "realworldqa",
            "method":            args.method,
            "compression_ratio": args.compression_ratio,
            "vision_weight":     args.vision_weight if args.method == "stratified_eviction" else None,
            "accuracy":          accuracy,
            "correct":           correct_count,
            "total":             total,
            "avg_prefill_ms":    total_prefill / total,
            "avg_decode_ms":     total_decode / total,
            "avg_peak_mem_gb":   total_mem / total,
            "avg_throughput":    total_throughput / total,
            "avg_img_tokens":    total_img_tokens / total,
        }, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
