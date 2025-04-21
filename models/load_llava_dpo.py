import torch
from peft import LoraConfig, PeftModel
from transformers import AutoModelForVision2Seq, AutoProcessor, LlavaForConditionalGeneration

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

model_path = "llava-hf/llava-1.5-7b-hf"

cache_dir = '/nlpgpu/data/gydou/cache'
base_model=LlavaForConditionalGeneration.from_pretrained(
    model_path,
    cache_dir=cache_dir,
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True,
    attn_implementation="flash_attention_2",
)

adapater_path = "/nlpgpu/data/gydou/cache/llava_dpo_4"
print("start loading model from", adapater_path)
model = PeftModel.from_pretrained(base_model, adapater_path)

# Inspect some LoRA delta weights before merge
# for name, param in model.named_parameters():
#     if "lora" in name or "adapter" in name:
#         print(f"{name}: {param.data.abs().mean().item():.6f}")

model = model.merge_and_unload()
print("model merged and unloaded")


processor = AutoProcessor.from_pretrained(
    model_path, trust_remote_code=True, do_image_splitting=False
)
tokenizer = processor.tokenizer

def call_model(image_file, prompt):
    conversation_1 = [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": image_file},
                {"type": "text", "text": prompt},
            ],
        },
    ]

    model.eval()
    inputs = processor.apply_chat_template(
        [conversation_1],
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        padding=True,
        return_tensors="pt"
    ).to(model.device, torch.float16)

    # Generate
    generate_ids = model.generate(**inputs, max_new_tokens=1500)
    raw_output = processor.batch_decode(generate_ids, skip_special_tokens=True)[0].strip()
    if "ASSISTANT:" in raw_output:
        return raw_output.split("ASSISTANT:", 1)[1].strip()
    return raw_output
