# model.py
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

def load_model():
    # load model in bfloat16; device_map="auto" places layers across available GPUs/CPU
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3-VL-4B-Instruct",
        dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")
    model.eval()
    return model, processor

def run_inference(model, processor, image, prompt, max_new_tokens=64, press=None):
    # format input as a chat turn with one image and one text message
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": prompt}
    ]}]

    # tokenize and apply the model's chat template; returns a dict of tensors
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt"
    )
    inputs.pop("token_type_ids", None)  # Qwen3-VL doesn't use token type ids
    inputs = inputs.to(model.device)

    input_seq_len = inputs["input_ids"].shape[1]

    # count image tokens by matching against the model's special image token id
    num_image_tokens = (inputs["input_ids"] == model.config.image_token_id).sum().item()

    torch.cuda.reset_peak_memory_stats()

    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)

    # prefill pass: generate exactly 1 token to isolate prefill latency
    start.record()
    with torch.no_grad():
        if press is not None:
            with press(model):  # apply KV cache compression if provided
                _ = model.generate(**inputs, max_new_tokens=1, do_sample=False)
        else:
            _ = model.generate(**inputs, max_new_tokens=1, do_sample=False)
    end.record()
    torch.cuda.synchronize()
    prefill_ms = start.elapsed_time(end)

    # full generation pass
    start.record()
    with torch.no_grad():
        if press is not None:
            with press(model):
                outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        else:
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    end.record()
    torch.cuda.synchronize()
    full_ms = start.elapsed_time(end)

    peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9
    decode_ms   = max(0.0, full_ms - prefill_ms)  # decode = full - prefill

    # strip the input tokens from the output to get only the generated tokens
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, outputs)]
    prediction  = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()

    num_output_tokens = len(trimmed[0])
    throughput = num_output_tokens / (full_ms / 1000)  # tok/s

    metrics = {
        "prefill_ms":        prefill_ms,
        "decode_ms":         decode_ms,
        "full_ms":           full_ms,
        "peak_mem_gb":       peak_mem_gb,
        "throughput_tokps":  throughput,
        "num_image_tokens":  num_image_tokens,
        "input_seq_len":     input_seq_len,
        "num_output_tokens": num_output_tokens,
    }

    return prediction, metrics
