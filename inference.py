import json, os
import sys
sys.path.append(('../'))
sys.path.append(('../../'))

# load data
import json
import argparse
import importlib
from utils.infer_on_data import *
# from models.load_LLaVA import *

mllm_to_module = {
    "gpt4": "load_GPT4o",
    "gpt4_multiagent": "load_GPT4_multiagent",
    "llava": "load_LLava_hf",
    "llava_dpo": "load_llava_dpo",
    "minigpt4": "load_MiniGPT4",
    "deepseek": "load_deepseek",
    "mplug": "load_mPLUG_Owl2",
    'qwenvl': 'load_Qwen_VL',
    'qwenvl_rl': 'load_Qwen_VL_rl',
    "gemini": "load_gemini",
    "claude": "load_claude",
}

# args
parser = argparse.ArgumentParser()
parser.add_argument("--mllm", type=str, default="qwenvl", choices=mllm_to_module.keys())
group = parser.add_mutually_exclusive_group()
group.add_argument(
    "--dpo",
    action="store_true",
    help="Use the DPO RL variant of the model"
)
group.add_argument(
    "--sft",
    action="store_true",
    help="Use the SFT RL variant of the model"
)
group.add_argument(
    "--sft_dpo",
    action="store_true",
    help="Use the SFT RL variant of the model"
)
parser.add_argument("--data_root", type=str, default='/nlpgpu/data/gydou/MSSBench_modified/mssbench')
parser.add_argument("--output_dir", type=str, default='/nlpgpu/data/gydou/MSSBench_modified/output/qwen_vl')
args = parser.parse_args()

rl_method="no_rl"
if args.dpo:
    rl_method = "dpo"
elif args.sft:
    rl_method = "sft"
elif args.sft_dpo:
    rl_method = "sft_dpo"
print(f"Loading {args.mllm} {rl_method} model..")
# Dynamic import based on mllm argument
module_name = f"models.{mllm_to_module[args.mllm]}"
model_module = importlib.import_module(module_name)
print(f"Module {module_name} imported successfully.")
print(f"Model module: {model_module}")
globals().update(vars(model_module))
 

print("Loading data..")
val_data = json.load(open(os.path.join(args.data_root, "combined_test.json"), 'r'))

# c_safe_acc, c_unsafe_acc, c_total_acc, e_safe_acc, e_unsafe_acc, e_total_acc = \
#     test_each_mss(val_data, call_model, args.data_root, output_path=os.path.join(args.output_dir, f"{args.mllm}_mssbench.json"))

print("Start running inference..")
test_each_mss(val_data, call_model, args.data_root, output_path=os.path.join(args.output_dir, f"{args.mllm}_{rl_method}_mssbench.json"))

with open(os.path.join(args.output_dir, f"{args.mllm}_{rl_method}_mssbench.json"), 'r') as f:
    responses = json.load(f)

# Make sure this is correct — adapt the filename if needed
save_file = os.path.join(args.output_dir, f"{args.mllm}_{rl_method}_mssbench_eval.json")

# Ensure directory exists for the eval file
os.makedirs(os.path.dirname(save_file), exist_ok=True)

print("Running evaluation..")
# Now directly call gpt4_eval on loaded responses
c_safe_acc, c_unsafe_acc, c_total_acc, e_safe_acc, e_unsafe_acc, e_total_acc = gpt4_eval(responses, save_file)


print(f"Chat Safe Acc: {c_safe_acc}, Chat Unsafe Acc: {c_unsafe_acc}, Chat Total Acc: {c_total_acc}")
print(f"Embodied Safe Acc: {e_safe_acc}, Embodied Unsafe Acc: {e_unsafe_acc}, Embodied Total Acc: {e_total_acc}")
