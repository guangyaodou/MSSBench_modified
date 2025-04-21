import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForVision2Seq, AutoProcessor, LlavaForConditionalGeneration
from PIL import Image
import requests

from transformers import (
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    GenerationConfig,
    is_comet_available,
)
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

model_path = "llava-hf/llava-1.5-7b-hf"

cache_dir = '/nlpgpu/data/gydou/cache'
model=LlavaForConditionalGeneration.from_pretrained(
    model_path,
    cache_dir=cache_dir,
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True,
    attn_implementation="flash_attention_2",
)

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


    # Turn on eval mode of the model
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
