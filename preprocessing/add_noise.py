import json
from datasets import load_from_disk

dataset_type = "comparisons"
dataset_path = "/media/sdebsarkar/extra-hdd/datasets/CSNLP/OpenAIHumanFeedback/{}".format(dataset_type)

dataset = load_from_disk(dataset_path)
print(len(dataset['train']))
# Function to convert dataset to JSON serializable format
def dataset_to_json(dataset):
    data = {}
    for split in dataset.keys():
        data[split] = dataset[split].to_dict()
    return data

dataset_json = dataset_to_json(dataset)['validation']
print(dataset_json.keys())
confidences = dataset_json['extra'][1000]
print(confidences)

# for split in dataset.keys():
#     print(f"\nSplit: {split}")
#     print(dataset[split].to_pandas().head())