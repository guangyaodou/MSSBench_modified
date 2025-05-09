import os
import torch
import wandb
import argparse
from datasets import load_dataset
from peft import LoraConfig, PeftModel
from transformers import Qwen2_5_VLForConditionalGeneration, AutoModelForVision2Seq, AutoProcessor, LlavaForConditionalGeneration


from transformers import (
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    GenerationConfig,
    is_comet_available,
)

from trl import (
    DPOConfig,
    DPOTrainer,
)
from PIL import Image

model_path_map = {
    "llava": "llava-hf/llava-1.5-7b-hf",
    "qwen": "Qwen/Qwen2.5-VL-7B-Instruct",
}

parser = argparse.ArgumentParser()
parser.add_argument("--mllm", type=str, default="qwen", choices=["llava", "qwen"])
parser.add_argument("--cache_dir", type=str, default="/nlpgpu/data/gydou/cache")
args = parser.parse_args()

rl_name = "sft_dpo"
model_path= model_path_map[args.mllm]
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    # load_in_4bit=True,
    # bnb_4bit_use_double_quant=True,
    # bnb_4bit_quant_type="nf4",
    # bnb_4bit_compute_dtype=torch.bfloat16
)

num_epochs=6

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    inference_mode=False,
    lora_dropout=0.01,
    bias="none",
    target_modules=['q_proj', 'v_proj', 'k_proj', 'o_proj'],
    task_type="CAUSAL_LM"
)

cache_dir = args.cache_dir

local_rank = int(os.environ.get("LOCAL_RANK", 0))

if local_rank == 0:
    wandb.init(
        project=f"{args.mllm}-{rl_name}",
        name=f"{args.mllm}-{rl_name}-{num_epochs}epochs",
        config={
            "model_path": model_path,
            "epochs": num_epochs,
            "learning_rate": 5e-5,
            "batch_size": 2,
            "gradient_accumulation_steps": 32,
            "beta": 0.1,
            "lora_r": 16,
            "lora_alpha": 32,
            "fp16": True,
        }
    )

device_map = {"": local_rank}
print("device_map", device_map)

if args.mllm == "llava":
    model=LlavaForConditionalGeneration.from_pretrained(
        model_path,
        cache_dir=cache_dir,
        device_map=local_rank,
        quantization_config=quantization_config,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(
        model_path, trust_remote_code=True, do_image_splitting=False
    )
    tokenizer = processor.tokenizer
elif args.mllm == "qwen":
    model=Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        cache_dir=cache_dir,
        device_map=local_rank,
        quantization_config=quantization_config,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(model_path)
    tokenizer = processor.tokenizer
else:
    raise ValueError(f"Unsupported model type: {args.mllm}")

############# Temporary fix for Qwen2.5-VL 7B model #############
adapater_path = f"/nlpgpu/data/gydou/cache/qwen_sft_6"
print("start loading model from", adapater_path)
model = PeftModel.from_pretrained(model, adapater_path)
model = model.merge_and_unload()
print("model merged and unloaded")

print("loading model done")


if lora_config is None:
    if args.mllm == "llava":
        ref_model=LlavaForConditionalGeneration.from_pretrained(
            model_path,
            cache_dir=cache_dir,
            device_map="auto",
            quantization_config=quantization_config,
            torch_dtype="auto",
            trust_remote_code=True,
        )
    elif args.mllm == "qwen":
        ref_model=Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            cache_dir=cache_dir,
            device_map="auto",
            quantization_config=quantization_config,
            torch_dtype="auto",
            trust_remote_code=True,
        )
else:
    ref_model=None


if model.config.model_type == "idefics2":
    pass  # the processor already has a valid chat template
elif model.config.model_type == "paligemma":
    processor.chat_template = """{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}<|im_start|>{% if message['role'] == 'user' %}USER: {% else %}ASSISTANT: {% endif %}{% for item in message['content'] if item['type'] == 'text' %}{{ item['text'] }}<|im_end|>{% endfor %}{% if message['role'] == 'user' %} {% else %}{{eos_token}}{% endif %}{% endfor %}{% if add_generation_prompt %}ASSISTANT: {% endif %}"""
elif model.config.model_type == "llava":
    processor.chat_template = """{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{% if message['role'] == 'user' %}USER: {% else %}ASSISTANT: {% endif %}{% for item in message['content'] %}{% if item['type'] == 'text' %}{{ item['text'] }}{% elif item['type'] == 'image' %}<image>{% endif %}{% endfor %}{% if message['role'] == 'user' %} {% else %}{{eos_token}}{% endif %}{% endfor %}{% if add_generation_prompt %}ASSISTANT: {% endif %}"""

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def resize_example(example):
    img = example["images"]
    if not isinstance(img, Image.Image):
        # if it's a path or bytes, open it
        img = Image.open(img)
    example["images"] = img.resize((224, 224))
    return example

train_dataset = load_dataset("gydou/mssbench_dpo", split="train")
val_dataset = load_dataset("gydou/mssbench_dpo", split="test")

# print number of examples in the train and val datasets
print("Initially, number of training examples:", len(train_dataset))
print("Initially, number of validation examples:", len(val_dataset))

phrase = "I am unable to provide a response to this request"

def keep_unable(example):
    # return True ⇾ keep, False ⇾ drop
    return isinstance(example["chosen"], str) and example["chosen"].startswith(phrase)

train_dataset = train_dataset.filter(keep_unable)

print("After filtering, number of training examples:", len(train_dataset))
print("After filtering, number of validation examples:", len(val_dataset))

train_dataset = train_dataset.map(resize_example)
val_dataset   = val_dataset.map(resize_example)

output_dir = f"/nlpgpu/data/gydou/output/{args.mllm}_{rl_name}_epoch_{num_epochs}"
if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

training_args = DPOConfig(output_dir=output_dir,
                          beta=0.1,
                          logging_steps=2,
                          dataset_num_proc=32,
                          per_device_train_batch_size=2,
                          gradient_checkpointing=False,
                          num_train_epochs=num_epochs,
                          save_steps=100,
                          learning_rate=1e-4,
                          gradient_accumulation_steps=32,
                          fp16=True,
                          report_to="wandb",
                          do_eval=True,
                          eval_strategy="steps",
                          eval_steps=2,
                          )

model.train()
trainer = DPOTrainer(
    model,
    ref_model=ref_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    processing_class=processor,
    peft_config=lora_config,
)

trainer.train()

wandb.finish()

save_dir = f"/nlpgpu/data/gydou/cache/{args.mllm}_{rl_name}_{num_epochs}"
if not os.path.exists(save_dir):
    os.makedirs(save_dir, exist_ok=True)
trainer.save_model(save_dir)
