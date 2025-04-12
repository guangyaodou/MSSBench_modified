import json
import csv
import os
import shutil

def create_csv_from_json():
    train_or_test = "test"
    # Load the JSON data
    with open(f"mssbench/dpo_dataset_{train_or_test}.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    # Specify the output CSV file name
    output_csv = f"dpo_mssbench/{train_or_test}/metadata.csv"

    # Write to CSV
    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["file_name", "prompt", "chosen", "rejected"])
        writer.writeheader()
        for entry in data:
            writer.writerow({
                "file_name": entry.get("image", ""),
                "prompt": entry.get("prompt", ""),
                "chosen": entry.get("chosen", ""),
                "rejected": entry.get("rejected", "")
            })


    print(f"CSV file has been written to {output_csv}")


def copy_image():
    train_or_test = "test"
    # Paths
    csv_path = f"dpo_mssbench/{train_or_test}/metadata.csv"  # path to your CSV
    src_image_folder = "mssbench/chat/"         # folder containing original images
    dst_image_folder = f"dpo_mssbench/{train_or_test}/"  # destination folder

    # Ensure destination directory exists
    os.makedirs(dst_image_folder, exist_ok=True)

    # Read the CSV and copy images
    with open(csv_path, "r", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            image_name = row["file_name"]
            src_path = os.path.join(src_image_folder, image_name)
            dst_path = os.path.join(dst_image_folder, image_name)

            if os.path.exists(src_path):
                shutil.copy2(src_path, dst_path)
                print(f"Copied: {image_name}")
            else:
                print(f"Missing: {image_name}")

if __name__ == "__main__":
    create_csv_from_json()