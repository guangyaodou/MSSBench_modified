import json
import random

# read json file from a given path
json_path = 'mssbench/combined.json'
with open(json_path, 'r') as f:
    data = json.load(f)

# split the data into two parts: chat
chat_data = data['chat']

# save random 70% of data as training data
random.shuffle(chat_data)
split_idx = int(0.7 * len(chat_data))

train_data = {'chat': chat_data[:split_idx]}
test_data = {'chat': chat_data[split_idx:]}

with open('mssbench/combined_train.json', 'w') as f:
    json.dump(train_data, f, indent=2)

with open('mssbench/combined_test.json', 'w') as f:
    json.dump(test_data, f, indent=2)
