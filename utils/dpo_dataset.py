import json
import os

dispreferred_safe = "I am unable to provide a response to this request because, while both the image and the question may appear safe on their own, their combination could result in an interpretation that leads to unsafe, misleading, or inappropriate outcomes. Ensuring safety in complex, real-world contexts often requires careful consideration of how different inputs interact, and in this case, responding without that caution could risk unintended consequences."

with open('output/gpt4_mssbench.json', 'r') as f:
    data_B = json.load(f)["chat"]
# Clean the image paths
for item in data_B:
    item["safe_image_path"] = os.path.basename(item["safe_img"])
    item["unsafe_image_path"] = os.path.basename(item["unsafe_img"])

with open('mssbench/combined_test.json', 'r') as f:
    train = json.load(f)['chat']

    # Create a mapping from (query, safe_image_path) to metadata for fast lookup
    metadata_lookup = {}
    for entry in train:
        for query in entry.get("queries", []):
            metadata_lookup[(query.strip(), entry["safe_image_path"])] = {
                "intent": entry.get("intent", "[unknown intent]"),
                "safe_image_path": entry.get("safe_image_path", "[unknown safe image path]"),
                "unsafe_image_path": entry.get("unsafe_image_path", "[unknown unsafe image path]"),
                "unsafe_image": entry.get("unsafe_image", "[unknown unsafe image]").lstrip(":") if isinstance(entry.get("unsafe_image"), str) else "[unknown unsafe image]",
                "reason": entry.get("reason", "no further explanation provided.")
            }

train_list = []
for item in train:
    train_list.append(item['safe_image_path'])
    train_list.append(item['unsafe_image_path'])

result = []

for item in data_B:
    full_prompt = item["prompt"]
    if "The user query is:" in full_prompt:
        parsed_prompt = full_prompt.split("The user query is:")[1].strip()
    else:
        parsed_prompt = full_prompt.strip()

    metadata = metadata_lookup.get((parsed_prompt, item["safe_image_path"]), None)
    if metadata:
        raw_reason = metadata['reason']
        clean_reason = raw_reason.lstrip(":* ").strip()

        if clean_reason != "no further explanation provided.":
            reason_text = f"The reason is {clean_reason} "
        else:
            reason_text = ""

        rejection_prompt = (
            f"I am unable to provide a response to this request because, while both the image and the question may appear safe on their own, "
            f"their combination - specifically, the unsafe image about {metadata['unsafe_image']} with the intent {metadata['intent']} - could lead to unsafe, misleading, or inappropriate outcomes. "
            f"{reason_text}Ensuring safety in complex, real-world contexts often requires careful consideration of how different inputs interact, "
            f"and in this case, responding without that caution could risk unintended consequences."
        )
        rejection_prompt = rejection_prompt.replace("unsafe image about :", "unsafe image about")
    else:
        rejection_prompt = dispreferred_safe

   # Safe image entry
    if item["safe_image_path"] in train_list:
        assert item["safe_image_path"] == metadata["safe_image_path"]
        result.append({
            "image": item["safe_image_path"],
            "prompt": parsed_prompt,
            "chosen": item["safe_output"].encode('utf-8').decode('unicode_escape').replace('\u2019', "'"),
            "rejected": dispreferred_safe
        })

    # Unsafe image entry
    if item["unsafe_image_path"] in train_list:
        assert item["unsafe_image_path"] == metadata["unsafe_image_path"]
        result.append({
            "image": item["unsafe_image_path"],
            "prompt": parsed_prompt,
            "chosen": rejection_prompt.encode('utf-8').decode('unicode_escape').replace('\u2019', "'"),
            "rejected": item["unsafe_output"].encode('utf-8').decode('unicode_escape').replace('\u2019', "'"),
        })

# Save to standard JSON file (valid JSON array)
with open("mssbench/dpo_dataset_test.json", "w") as f:
    json.dump(result, f, indent=2)
print(f"Total number of entries: {len(result)}")
