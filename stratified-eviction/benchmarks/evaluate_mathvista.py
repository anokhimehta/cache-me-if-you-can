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


def extract_number(text):
    matches = re.findall(r'-?\d+\.?\d*', text)
    return matches[-1] if matches else text.strip()


def is_correct(prediction, correct, question_type):
    if question_type == "multi_choice":
        return correct.lower() in prediction.lower()
    pred_num = extract_number(prediction)
    try:
        return abs(float(pred_num) - float(correct)) < 0.1
    except Exception:
        return pred_num.strip() == correct.strip()


def main():
    args = parse_args()

    if args.method is not None:
        mod = importlib.import_module(f"methods.{args.method}")
        press = mod.get_press(compression_ratio=args.compression_ratio, vision_weight=args.vision_weight)
    else:
        press = None

    model, processor = load_model()
    ds = load_dataset("AI4Math/MathVista", split="testmini")

    correct_count    = 0
    total            = 0
    total_prefill    = 0
    total_decode     = 0
    total_mem        = 0
    total_throughput = 0
    total_img_tokens = 0

    for sample in ds:
        image         = sample["decoded_image"]
        query         = sample["query"] + "\nGive only the final answer, nothing else."
        correct       = sample["answer"]
        question_type = sample["question_type"]

        prediction, metrics = run_inference(model, processor, image, query, max_new_tokens=64, press=press)

        total_prefill    += metrics["prefill_ms"]
        total_decode     += metrics["decode_ms"]
        total_mem        += metrics["peak_mem_gb"]
        total_throughput += metrics["throughput_tokps"]
        total_img_tokens += metrics["num_image_tokens"]

        correct_flag = is_correct(prediction, correct, question_type)
        correct_count += correct_flag
        total += 1

        print(f"[{total}] Type: {question_type}")
        print(f"  Predicted : {prediction}")
        print(f"  Correct   : {correct}")
        print(f"  {'✓' if correct_flag else '✗'}")

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
    out_path = os.path.join("results", f"mathvista_{tag}.json")
    with open(out_path, "w") as f:
        json.dump({
            "benchmark":         "mathvista",
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
