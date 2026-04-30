#!/usr/bin/env python3
import os
import re
import json
import math
import time
from typing import Any, Dict, Optional, Tuple
import sys
sys.path.append("/scratch/al9581/project_work/vision-zip")
import torch
# from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from transformers import AutoProcessor
from modeling_qwen3_vl import Qwen3VLForConditionalGeneration


MODEL_NAME = "Qwen/Qwen3-VL-4B-Instruct"


def load_model_and_processor(
    model_name: str = MODEL_NAME,
    device_map: str = "auto",
    torch_dtype: str = "auto",
    attn_implementation: Optional[str] = None,
):
    dtype_map = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }

    kwargs = {
        "pretrained_model_name_or_path": model_name,
        "device_map": device_map,
        "torch_dtype": dtype_map[torch_dtype],
    }
    if attn_implementation is not None:
        kwargs["attn_implementation"] = attn_implementation

    model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_name)
    return model, processor


'''
The different methods of dropping:
1. dominant : keep top k important visual tokens and then compress the remaining ones into fewer contextual tokens
2. clustering : local cluster aggregation instead of global residual averaging.

'''




def generate_answer(
    model,
    processor,
    image,
    prompt: str,
    max_new_tokens: int = 32,
    do_sample: bool = False,
    temperature: float = 0.0,
    num_beams: int = 1,
    visionzip_keep_ratio : Optional[float] = None, 
    visionzip_mode : str = "dominant",
) -> Tuple[str, Dict[str, Any]]:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
############################################
    input_shapes = {
        k : tuple(v.shape) if hasattr(v,"shape") else str(type(v))
        for k, v in inputs.items()
    }
        # =========================
    # VisionZip (DROP MODE)
    # =========================
    if visionzip_keep_ratio is not None and "pixel_values" in inputs:
    
        X = inputs["pixel_values"]   # (N, D)
        N = X.shape[0]
        K = max(1, int(N * visionzip_keep_ratio))
    
        # importance score: L2 norm
        scores = torch.norm(X, dim=1)
    
        # top-K tokens
        topk = torch.topk(scores, K).indices

        keep_mask = torch.zeros(N, device=X.device, dtype=torch.bool)
        keep_mask[topk] = True

        X_masked = X.clone()
        X_masked[~keep_mask] = 0
        inputs["pixel_values"] = X_masked
       
        image_tokens_before = N
        image_tokens_after = K
    else:
        image_tokens_before = None
        image_tokens_after = None


    if not hasattr(generate_answer, "_printed_shapes"):
        print("INPUT SHAPES:", input_shapes)
        generate_answer._printed_shapes = True



    
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    start = time.time()
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            num_beams=num_beams,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    latency = time.time() - start

    trimmed = []
    total_generated = 0
    for in_ids, out_ids in zip(inputs["input_ids"], generated_ids):
        new_ids = out_ids[len(in_ids):]
        trimmed.append(new_ids)
        total_generated += len(new_ids)

    text = processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    peak_memory_gb = None
    if torch.cuda.is_available():
        peak_memory_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

        # Prefill (forward pass)
    prefill_start = time.time()
    with torch.no_grad():
        _ = model(**inputs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    prefill_time = time.time() - prefill_start


    decode_start = time.time()
    with torch.no_grad():
        generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        num_beams=num_beams,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    decode_time = time.time() - decode_start

    metrics = {
        "latency_sec": latency,
        "generated_tokens": total_generated,
        "peak_memory_gb": peak_memory_gb,
        "throughput_tok_per_sec": total_generated / latency if latency > 0 else None,
        "prefill_sec": prefill_time,
        "decode_sec": decode_time,
        "input_shapes": input_shapes,
        "image_tokens_before": image_tokens_before,
        "image_tokens_after": image_tokens_after,
    }
    return text, metrics


def normalize_text(x: Any) -> str:
    if x is None:
        return ""
    x = str(x).strip().lower()
    x = re.sub(r"\s+", " ", x)
    return x


def normalize_simple_answer(x: Any) -> str:
    x = normalize_text(x)
    x = re.sub(r"[^\w\s\.\-/%]", "", x)
    return x.strip()


def extract_last_line(text: str) -> str:
    lines = [line.strip() for line in str(text).splitlines() if line.strip()]
    return lines[-1] if lines else str(text).strip()


def extract_after_marker(text: str) -> str:
    s = str(text).strip()
    patterns = [
        r"final answer\s*[:：]\s*(.+)",
        r"answer\s*[:：]\s*(.+)",
        r"答案\s*[:：]\s*(.+)",
    ]
    for pat in patterns:
        m = re.search(pat, s, flags=re.IGNORECASE | re.DOTALL)
        if m:
            return m.group(1).strip()
    return extract_last_line(s)


def extract_first_number(text: Any) -> Optional[float]:
    s = str(text).replace(",", "")
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    if not m:
        return None
    return float(m.group(0))


def extract_last_number(text: Any) -> Optional[float]:
    s = str(text).replace(",", "")
    matches = re.findall(r"-?\d+(?:\.\d+)?", s)
    if not matches:
        return None
    return float(matches[-1])


def numeric_equal(a: Any, b: Any, rel_tol: float = 1e-4, abs_tol: float = 1e-4) -> bool:
    try:
        return math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=abs_tol)
    except Exception:
        return False


def save_results(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)