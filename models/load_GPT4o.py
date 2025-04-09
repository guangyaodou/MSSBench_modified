import sys
sys.path.append('../')
sys.path.append('../../')
import os
import json
import base64
import requests
from mimetypes import guess_type
from openai import OpenAI
from PIL import Image
import base64
from io import BytesIO

# Set OpenAI API Key from environment variable
openai_api_key = os.getenv("OPENAI_API_KEY")

# OpenAI model name (corresponds to Azure's "deployment_name")
model_name = "gpt-4o-mini"  # Or whatever model you're using (adjust if needed)

# Base URL for OpenAI (direct)
openai_api_base = "https://api.openai.com/v1"

# Function to convert local image to base64 data URL
# def local_image_to_data_url(image_path):
#     mime_type, _ = guess_type(image_path)
#     if mime_type is None:
#         mime_type = 'application/octet-stream'
#
#     with open(image_path, "rb") as image_file:
#         base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')
#
#     return f"data:{mime_type};base64,{base64_encoded_data}"
#
# # Function to call the OpenAI model (same behavior as original)
# def call_model(image_path, prompt):
#     try:
#         headers = {
#             "Content-Type": "application/json",
#             "Authorization": f"Bearer {openai_api_key}"
#         }
#
#         payload = {
#             "model": model_name,
#             "messages": [
#                 {
#                     "role": "user",
#                     "content": [
#                         {
#                             "type": "text",
#                             "text": prompt
#                         },
#                         {
#                             "type": "image_url",
#                             "image_url": {
#                                 "url": local_image_to_data_url(image_path)
#                             }
#                         }
#                     ]
#                 }
#             ],
#             "max_tokens": 1500
#         }
#
#         response = requests.post(f"{openai_api_base}/chat/completions", headers=headers, json=payload)
#
#         # Parse response (same logic as original)
#         response_json = response.json()
#         # print("testtest response_json", response_json, "\n")
#         return response_json['choices'][0]['message']['content']
#
#     except Exception as e:
#         return f"exception {e}, and return None"

# Set up client
client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)
def local_image_to_data_url(image_path):
    with Image.open(image_path) as img:
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/png;base64,{img_str}"

# Main function
def call_model(image_path, prompt):
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": local_image_to_data_url(image_path)
                            }
                        }
                    ]
                }
            ],
            max_tokens=1500,
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"exception {e}, and return None"
