import json, os
import sys

sys.path.append(('../'))
sys.path.append(('../../'))

# load data
import json
import argparse
import importlib
from utils.batch_api_calls_utils import *
# from utils.infer_on_data import *

# from models.load_LLaVA import *

mllm_to_module = {
    "gpt4": "load_GPT4o",
    "llava": "load_LLaVA",
    "minigpt4": "load_MiniGPT4",
    "deepseek": "load_deepseek",
    "mplug": "load_mPLUG_Owl2",
    'qwenvl': 'load_Qwen_VL',
    "gemini": "load_gemini",
    "claude": "load_claude",
}

# args
parser = argparse.ArgumentParser()
parser.add_argument("--mllm", type=str, default="llava", choices=mllm_to_module.keys())
parser.add_argument("--data_root", type=str, default='/Users/xinyuandou/Desktop/MSSBench_modified/mssbench')
parser.add_argument("--output_dir", type=str, default='/Users/xinyuandou/Desktop/MSSBench_modified/output')
args = parser.parse_args()

val_data = json.load(open(os.path.join(args.data_root, "combined.json"), 'r'))

generate_batch_jsonl(data=val_data, img_root=args.data_root, output_jsonl_path=os.path.join(args.output_dir, f"batch/{args.mllm}_batch.jsonl"))


def reconstruct_outputs(result_jsonl_path, request_metadata):
    outputs = {"chat": [], "embodied": []}
    grouped = {}

    with open(result_jsonl_path, "r") as f:
        for line in f:
            result = json.loads(line)
            custom_id = result["custom_id"]
            response = result["response"]

            if "choices" not in response or not response["choices"]:
                print(f"[Warning] No choices found for {custom_id}")
                continue

            content = response["choices"][0]["message"]["content"]
            meta = request_metadata[custom_id]
            key = (meta["type"], meta["i"], meta["j"])

            if key not in grouped:
                grouped[key] = {
                    "prompt": meta["prompt"],
                    "safe_img": None,
                    "unsafe_img": None,
                    "safe_output": None,
                    "unsafe_output": None
                }

            grouped[key][f"{meta['tag']}_img"] = meta["image_path"]
            grouped[key][f"{meta['tag']}_output"] = content

    # Final formatting into outputs['chat'] and outputs['embodied']
    for (type_, _, _), entry in grouped.items():
        outputs[type_].append(entry)

    return outputs


# test_each_mss(val_data, call_model, args.data_root,
#               output_path=os.path.join(args.output_dir, f"{args.mllm}_mssbench.json"))
#
# with open(os.path.join(args.output_dir, f"{args.mllm}_mssbench.json"), 'r') as f:
#     responses = json.load(f)
#
# # Make sure this is correct — adapt the filename if needed
# save_file = os.path.join(args.output_dir, f"{args.mllm}_mssbench_eval.json")
#
# # Ensure directory exists for the eval file
# os.makedirs(os.path.dirname(save_file), exist_ok=True)
#
# # Now directly call gpt4_eval on loaded responses
# c_safe_acc, c_unsafe_acc, c_total_acc, e_safe_acc, e_unsafe_acc, e_total_acc = gpt4_eval(responses, save_file)
#
# print(f"Chat Safe Acc: {c_safe_acc}, Chat Unsafe Acc: {c_unsafe_acc}, Chat Total Acc: {c_total_acc}")
# print(f"Embodied Safe Acc: {e_safe_acc}, Embodied Unsafe Acc: {e_unsafe_acc}, Embodied Total Acc: {e_total_acc}")
