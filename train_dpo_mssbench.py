import os
import torch
import wandb
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForVision2Seq, AutoProcessor, LlavaForConditionalGeneration

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

quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    # load_in_4bit=True,
    # bnb_4bit_use_double_quant=True,
    # bnb_4bit_quant_type="nf4",
    # bnb_4bit_compute_dtype=torch.bfloat16
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    inference_mode=False,
    lora_dropout=0.01,
    bias="none",
    target_modules=['q_proj', 'v_proj', 'k_proj', 'o_proj'],
    task_type="CAUSAL_LM"
)

model_path = "llava-hf/llava-1.5-7b-hf"
cache_dir = '/nlpgpu/data/gydou/cache'

wandb.init(
    project="llava-dpo",
    name="llava-dpo-10epochs",
    config={
        "model_path": model_path,
        "epochs": 10,
        "learning_rate": 5e-5,
        "batch_size": 2,
        "gradient_accumulation_steps": 32,
        "beta": 0.1,
        "lora_r": 16,
        "lora_alpha": 32,
        "fp16": True,
    }
)

local_rank = int(os.environ.get("LOCAL_RANK", 0))
device_map = {"": local_rank}
print("device_map", device_map)

model=LlavaForConditionalGeneration.from_pretrained(
    model_path,
    cache_dir=cache_dir,
    # device_map="balanced",
    device_map=local_rank,
    quantization_config=quantization_config,
    torch_dtype=torch.float16,
    trust_remote_code=True,
)

if lora_config is None:
    ref_model=LlavaForConditionalGeneration.from_pretrained(
        model_path,
        cache_dir=cache_dir,
        device_map="auto",
        quantization_config=quantization_config,
        torch_dtype="auto",
        trust_remote_code=True,
    )
else:
    ref_model=None

processor = AutoProcessor.from_pretrained(
    model_path, trust_remote_code=True, do_image_splitting=False
)
tokenizer = processor.tokenizer

print("loading model done")

if model.config.model_type == "idefics2":
    pass  # the processor already has a valid chat template
elif model.config.model_type == "paligemma":
    processor.chat_template = """{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}<|im_start|>{% if message['role'] == 'user' %}USER: {% else %}ASSISTANT: {% endif %}{% for item in message['content'] if item['type'] == 'text' %}{{ item['text'] }}<|im_end|>{% endfor %}{% if message['role'] == 'user' %} {% else %}{{eos_token}}{% endif %}{% endfor %}{% if add_generation_prompt %}ASSISTANT: {% endif %}"""
elif model.config.model_type == "llava":
    processor.chat_template = """{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{% if message['role'] == 'user' %}USER: {% else %}ASSISTANT: {% endif %}{% for item in message['content'] %}{% if item['type'] == 'text' %}{{ item['text'] }}{% elif item['type'] == 'image' %}<image>{% endif %}{% endfor %}{% if message['role'] == 'user' %} {% else %}{{eos_token}}{% endif %}{% endfor %}{% if add_generation_prompt %}ASSISTANT: {% endif %}"""

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

train_dataset = load_dataset("gydou/mssbench_dpo", split="train")
val_dataset = load_dataset("gydou/mssbench_dpo", split="test")

output_dir = "/nlpgpu/data/gydou/output/Llava_DPO_epoch_10"
if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

training_args = DPOConfig(output_dir=output_dir,
                          beta=0.1,
                          logging_steps=10,
                          dataset_num_proc=32,
                          per_device_train_batch_size=2,
                          gradient_checkpointing=False,
                          num_train_epochs=10,
                          save_steps=100,
                          learning_rate=5e-5,
                          gradient_accumulation_steps =32,
                          fp16=True,
                          report_to="wandb",
                          )

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

save_dir = "/nlpgpu/data/gydou/cache/llava_dpo_10"
if not os.path.exists(save_dir):
    os.makedirs(save_dir, exist_ok=True)
trainer.save_model(save_dir)
wandb.log({"final_loss": trainer.state.loss})
wandb.finish()
