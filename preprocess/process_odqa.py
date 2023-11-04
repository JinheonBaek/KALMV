import os
import json

from datasets import load_dataset

data_to_splits = {
    'nq_open': ['train', 'validation'],
    'hotpot_qa': ['train', 'validation']
}

def get_question(data_name, sample):
    if data_name == 'nq_open':
        return sample['question'] + "?"
    if data_name == 'hotpot_qa':
        return sample['question']

    raise ValueError

def get_answer(data_name, sample):
    if data_name == 'nq_open':
        return [
            {
                "name": data_name.upper(),
                "mention": answer,
                "span": None,
                "description": None
            } for answer in sample['answer']
        ]
    if data_name == 'hotpot_qa':
        return [
            {
                "name": data_name.upper(),
                "mention": sample['answer'],
                "span": None,
                "description": None
            }
        ]

    raise ValueError

def convert_orign_to_jsonl(data_name, data_splits, orign_datasets):
    jsonl_datasets = {split: [] for split in data_splits}

    for split in data_splits:
        for sample in orign_datasets[split]:
            jsonl_datasets[split].append({
                "question": get_question(data_name, sample),
                "answer_entities": get_answer(data_name, sample)
            })

    return jsonl_datasets

def save_jsonl_data(data_name, data_splits, output_data_path, jsonl_datasets):
    output_dir = os.path.join(output_data_path, data_name)
    if not os.path.exists(output_dir):  os.makedirs(output_dir)

    for split in data_splits:
        dataset = jsonl_datasets[split]
        
        with open(
            os.path.join(output_dir, f"{split if split == 'train' else 'test'}-samples.jsonl"), 'w'
        ) as outfile:
            for sample in dataset:
                outfile.write(json.dumps(sample) + '\n')

for data_name in ['nq_open', 'hotpot_qa']:
    data_splits = data_to_splits[data_name]
    output_data_path = './datasets'
    orign_datasets = load_dataset(data_name) if data_name == 'nq_open' else load_dataset(data_name, 'fullwiki')
    jsonl_datasets = convert_orign_to_jsonl(data_name, data_splits, orign_datasets)
    save_jsonl_data(data_name, data_splits, output_data_path, jsonl_datasets)
