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
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    get_model_name_from_path,
)

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
)


lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    inference_mode=False,
    lora_dropout=0.01,
    bias="none",
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM"
)

disable_torch_init()
model_path = "liuhaotian/llava-v1.6-vicuna-7b"
model_name = get_model_name_from_path(model_path)

cache_dir = '/nlpgpu/data/gydou/cache'
model=LlavaForConditionalGeneration.from_pretrained(
    model_path,
    cache_dir=cache_dir,
    device_map="auto",
    quantization_config=quantization_config,
    torch_dtype="auto",
    trust_remote_code=True,
)

ref_model=LlavaForConditionalGeneration.from_pretrained(
    model_path,
    cache_dir=cache_dir,
    device_map="auto",
    quantization_config=quantization_config,
    torch_dtype="auto",
    trust_remote_code=True,
)

processor = AutoProcessor.from_pretrained(
    model_path, trust_remote_code=True, do_image_splitting=False
)
tokenizer = processor.tokenizer

# tokenizer, model, processor, context_len = load_pretrained_model(
#     model_path, None, model_name, cache_dir=cache_dir, device_map="auto", quantization_config=quantization_config,
# )
#
# _, ref_model, _, _ = load_pretrained_model(
#     model_path, None, model_name, cache_dir=cache_dir, device_map="auto", quantization_config=quantization_config,
# )

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

training_args = DPOConfig(output_dir="/nlpgpu/data/gydou/output/Llava_DPO",
                          logging_steps=10,
                          per_device_train_batch_size=4,
                          num_train_epochs=5,
                          save_steps=100,
                          learning_rate=5e-5,
                          fp16=True
                          )

trainer = DPOTrainer(
    model,
    ref_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    processing_class=processor,
    peft_config=lora_config,
)

trainer.train()

# Save and push to hub
trainer.save_model("/nlpgpu/data/gydou/cache/llava_dpo")
