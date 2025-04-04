import os
import json
import base64
from PIL import Image
from io import BytesIO
from mimetypes import guess_type
from tqdm import tqdm

from utils.prompts import *

MODEL_NAME = "gpt-4o-mini"

def local_image_to_data_url(image_path):
    with Image.open(image_path) as img:
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/png;base64,{img_str}"

def generate_batch_jsonl(data, img_root, output_jsonl_path):
    batch_lines = []
    request_metadata = {}

    # Process chat samples
    for i, d in tqdm(enumerate(data['chat']), desc="Processing chat"):
        safe_img = os.path.join(img_root, 'chat', d['safe_image_path'])
        unsafe_img = os.path.join(img_root, 'chat', d['unsafe_image_path'])

        for j, query in enumerate(d['queries']):
            for tag, image_path in zip(["safe", "unsafe"], [safe_img, unsafe_img]):
                prompt = PROMPT_CHAT_IF + query
                try:
                    data_url = local_image_to_data_url(image_path)
                except Exception as e:
                    print(f"[Warning] Failed to load image: {image_path}, skipping.")
                    continue

                custom_id = f"chat_{i}_{j}_{tag}"
                request = {
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": MODEL_NAME,
                        "messages": [{
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": data_url}}
                            ]
                        }],
                        "max_tokens": 1000
                    }
                }

                batch_lines.append(request)
                request_metadata[custom_id] = {
                    "type": "chat",
                    "i": i,
                    "j": j,
                    "tag": tag,
                    "prompt": prompt,
                    "image_path": image_path
                }

    # Process embodied samples
    for i, d in tqdm(enumerate(data['embodied']), desc="Processing embodied"):
        safe_img = os.path.join(img_root, "embodied", d['safe'])
        unsafe_img = os.path.join(img_root, "embodied", d['unsafe'])

        for j, (safe_instr, unsafe_instr) in enumerate(zip(d['safe_instructions'], d['unsafe_instructions'])):
            for tag, instr, image_path in zip(["safe", "unsafe"], [safe_instr, unsafe_instr], [safe_img, unsafe_img]):
                prompt = PROMPT_EMBODIED_IF + instr
                try:
                    data_url = local_image_to_data_url(image_path)
                except Exception as e:
                    print(f"[Warning] Failed to load image: {image_path}, skipping.")
                    continue

                custom_id = f"embodied_{i}_{j}_{tag}"
                request = {
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": MODEL_NAME,
                        "messages": [{
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": data_url}}
                            ]
                        }],
                        "max_tokens": 1000
                    }
                }

                batch_lines.append(request)
                request_metadata[custom_id] = {
                    "type": "embodied",
                    "i": i,
                    "j": j,
                    "tag": tag,
                    "prompt": prompt,
                    "image_path": image_path
                }

    with open(output_jsonl_path, "w") as f:
        for line in batch_lines:
            f.write(json.dumps(line) + "\n")

    print(f"✅ Batch input saved to: {output_jsonl_path}")
    return request_metadata
