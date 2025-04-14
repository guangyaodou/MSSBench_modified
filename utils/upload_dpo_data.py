from PIL import Image
from datasets import load_dataset
from datasets import Image as HFImage

dataset = load_dataset("imagefolder", data_dir="./dpo_mssbench")

dataset = dataset.rename_column("image", "images")

dataset.push_to_hub("gydou/mssbench_dpo", private=True)

# dataset_train = load_dataset("gydou/mssbench_dpo", split="train")
#
# print("Column names:", dataset_train.column_names)
#
# dataset_train = dataset_train.cast_column("image", HFImage())
#
# # Show first 10 images
# for i in range(10):
#     dataset_train[i]["image"].show()