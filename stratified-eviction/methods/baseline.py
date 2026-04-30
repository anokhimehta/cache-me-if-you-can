#code to implement baseline model of Qwen3-VL-4B-Instruct
import torch
from PIL import Image
from model import load_model

image = Image.open("dog.jpg").convert("RGB")

model, processor = load_model()

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "What is in this image?"}
        ]
    }
]

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
)
inputs.pop("token_type_ids", None)
inputs = inputs.to(model.device)

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=128)

trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, outputs)]
answer = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

print("Answer:", answer)