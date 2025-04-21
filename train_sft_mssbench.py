import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import wandb
import argparse
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import Qwen2_5_VLForConditionalGeneration, AutoModelForVision2Seq, AutoProcessor, LlavaForConditionalGeneration, Qwen2VLProcessor
from qwen_vl_utils import process_vision_info
from datasets import Dataset

from utils.prompts import *
import gc
import time

from transformers import (
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    GenerationConfig,
    is_comet_available,
)

from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)

parser = argparse.ArgumentParser()
parser.add_argument("--mllm", type=str, default="qwen", choices=["llava", "qwen"])
parser.add_argument("--cache_dir", type=str, default="/nlpgpu/data/gydou/cache")
args = parser.parse_args()

def format_data(sample):
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": sample["images"],
                },
                {
                    "type": "text",
                    "text": PROMPT_CHAT_IF + sample["prompt"],
                },
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": sample["chosen"]}],
        },
    ]


train_dataset = load_dataset("gydou/mssbench_sft_2", split="train")
val_dataset = load_dataset("gydou/mssbench_sft_2", split="test")

train_dataset = [format_data(sample) for sample in train_dataset]
val_dataset = [format_data(sample) for sample in val_dataset]

def clear_memory():
    # Delete variables if they exist in the current global scope
    if "inputs" in globals():
        del globals()["inputs"]
    if "model" in globals():
        del globals()["model"]
    if "processor" in globals():
        del globals()["processor"]
    if "trainer" in globals():
        del globals()["trainer"]
    if "peft_model" in globals():
        del globals()["peft_model"]
    if "bnb_config" in globals():
        del globals()["bnb_config"]
    time.sleep(2)

    # Garbage collection and clearing CUDA memory
    gc.collect()
    time.sleep(2)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    time.sleep(2)
    gc.collect()
    time.sleep(2)

    print(f"GPU allocated memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"GPU reserved memory: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")



clear_memory()


model_path_map = {
    "llava": "llava-hf/llava-1.5-7b-hf",
    "qwen": "Qwen/Qwen2.5-VL-7B-Instruct",
}

parser = argparse.ArgumentParser()
parser.add_argument("--mllm", type=str, default="qwen", choices=["llava", "qwen"])
parser.add_argument("--cache_dir", type=str, default="/nlpgpu/data/gydou/cache")
args = parser.parse_args()

rl_name = "sft"
model_path= model_path_map[args.mllm]
quantization_config = BitsAndBytesConfig(
    # load_in_8bit=True,
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

num_epochs=6

cache_dir = args.cache_dir

local_rank = int(os.environ.get("LOCAL_RANK", 0))

if local_rank == 0:
    wandb.init(
        project=f"{args.mllm}-{rl_name}",
        name=f"{args.mllm}-{rl_name}-{num_epochs}epochs",
        config={
            "model_path": model_path,
            "epochs": num_epochs,
            "learning_rate": 1e-5,
            "batch_size": 1,
            "gradient_accumulation_steps": 8,
            "beta": 0.1,
            "lora_r": 8,
            "lora_alpha": 16,
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
    processor = Qwen2VLProcessor.from_pretrained(model_path)
    tokenizer = processor.tokenizer
else:
    raise ValueError(f"Unsupported model type: {args.mllm}")

model.config.use_cache = False
print("loading model done")

if model.config.model_type == "idefics2":
    pass  # the processor already has a valid chat template
elif model.config.model_type == "paligemma":
    processor.chat_template = """{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}<|im_start|>{% if message['role'] == 'user' %}USER: {% else %}ASSISTANT: {% endif %}{% for item in message['content'] if item['type'] == 'text' %}{{ item['text'] }}<|im_end|>{% endfor %}{% if message['role'] == 'user' %} {% else %}{{eos_token}}{% endif %}{% endfor %}{% if add_generation_prompt %}ASSISTANT: {% endif %}"""
elif model.config.model_type == "llava":
    processor.chat_template = """{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{% if message['role'] == 'user' %}USER: {% else %}ASSISTANT: {% endif %}{% for item in message['content'] %}{% if item['type'] == 'text' %}{{ item['text'] }}{% elif item['type'] == 'image' %}<image>{% endif %}{% endfor %}{% if message['role'] == 'user' %} {% else %}{{eos_token}}{% endif %}{% endfor %}{% if add_generation_prompt %}ASSISTANT: {% endif %}"""


peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    inference_mode=False,
    lora_dropout=0.01,
    bias="none",
    target_modules=['q_proj', 'v_proj'],
    task_type="CAUSAL_LM"
)

def collate_fn(examples):
    # Get the texts and images, and apply the chat template
    texts = [
        processor.apply_chat_template(example, tokenize=False) for example in examples
    ]  # Prepare texts for processing
    # image_inputs = [process_vision_info(example)[0] for example in examples]
    image_inputs = [img.resize((224, 224)) for example in examples for img in process_vision_info(example)[0]]  # Process the images to extract inputs

    # Tokenize the texts and process the images
    batch = processor(
        text=texts, images=image_inputs, return_tensors="pt", padding=True
    )  # Encode texts and images into tensors

    # The labels are the input_ids, and we mask the padding tokens in the loss computation
    labels = batch["input_ids"].clone()  # Clone input IDs for labels
    labels[labels == processor.tokenizer.pad_token_id] = -100  # Mask padding tokens in labels

    # Ignore the image token index in the loss computation (model specific)
    if isinstance(processor, Qwen2VLProcessor):  # Check if the processor is Qwen2VLProcessor
        image_tokens = [151652, 151653, 151655]  # Specific image token IDs for Qwen2VLProcessor
    else:
        image_tokens = [processor.tokenizer.convert_tokens_to_ids(processor.image_token)]  # Convert image token to ID

    # Mask image token IDs in the labels
    for image_token_id in image_tokens:
        labels[labels == image_token_id] = -100  # Mask image token IDs in labels

    batch["labels"] = labels  # Add labels to the batch

    return batch  # Return the prepared batch




if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# raw_train = load_dataset("gydou/mssbench_sft_2", split="train")
# train_list = [ {"messages": format_data(sample)} for sample in raw_train ]
# train_dataset = Dataset.from_list(train_list)
#
# raw_val = load_dataset("gydou/mssbench_sft_2", split="test")
# val_list = [ {"messages": format_data(sample)} for sample in raw_val ]
# val_dataset = Dataset.from_list(val_list)



output_dir = f"/nlpgpu/data/gydou/output/{args.mllm}_{rl_name}_epoch_{num_epochs}"
if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

training_args = SFTConfig(output_dir=output_dir,
                          logging_steps=2,
                          dataset_num_proc=32,
                          dataset_kwargs={"skip_prepare_dataset": True},
                          remove_unused_columns=False,
                          per_device_train_batch_size=1,
                          per_device_eval_batch_size=1,
                          gradient_checkpointing=False,
                          # gradient_checkpointing_kwargs={"use_reentrant": False},
                          num_train_epochs=num_epochs,
                          save_steps=20,
                          learning_rate=1e-5,
                          lr_scheduler_type="constant",
                          gradient_accumulation_steps=8,
                          fp16=True,
                          tf32=True,
                          max_grad_norm=0.3,
                          warmup_ratio=0.03,
                          report_to="wandb",
                          do_eval=True,
                          eval_strategy="steps",
                          eval_steps=2,
                          )

trainer = SFTTrainer(
    model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    processing_class=processor.tokenizer,
    data_collator=collate_fn,
    peft_config=peft_config,
)

torch.cuda.empty_cache()

trainer.train()

wandb.finish()

save_dir = f"/nlpgpu/data/gydou/cache/{args.mllm}_{rl_name}_{num_epochs}"
if not os.path.exists(save_dir):
    os.makedirs(save_dir, exist_ok=True)
trainer.save_model(save_dir)
