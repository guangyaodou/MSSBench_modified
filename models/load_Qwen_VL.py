from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
import json
import tqdm
import random
from PIL import Image
import os

torch.manual_seed(1234)
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"



# Function to generate caption with grounding
def call_model(image_path, text_prompt):
    model_path = "Qwen/Qwen2.5-VL-7B-Instruct"
    cache_dir = '/nlpgpu/data/gydou/cache'

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype="auto", device_map="auto", trust_remote_code=True, cache_dir=cache_dir
    )

    processor = AutoProcessor.from_pretrained(model_path)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                },
                {"type": "text", "text": text_prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return output_text[0].strip()
