import os
import json

from typing import Dict
from multiprocessing import cpu_count

import torch
from torch.utils.data import Dataset, DataLoader

FOLDER_NAMES = {
    'NaturalQuestions': 'nq_open',
    'HotpotQA': 'hotpot_qa'
}

class QADataset(Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, ind: int):
        return self.data[ind]

    def __iter__(self):
        for sample in self.data:
            yield sample

    @staticmethod
    def collate(x):
        return x

def get_qa_datasets(args, cpu_usage_ratio=1.0):
    datasets = {
        split: QADataset([
            json.loads(sample) for sample in \
            list(open(os.path.join(args.data_path, f'{FOLDER_NAMES[args.data_name]}', f'{split}-samples.jsonl'), 'r'))
        ])
        for split in args.data_splits
    }
    data_loaders = {
        split: DataLoader(
            datasets[split],
            batch_size = args.batch_size,
            shuffle = split == 'train' and not isinstance(dataset, torch.utils.data.IterableDataset),
            num_workers = int(cpu_count() * cpu_usage_ratio),
            collate_fn = datasets[split].collate
        )
        for split, dataset in datasets.items()
    }
    return datasets, data_loaders


class KG(object):
    def __init__(self, args):
        super(KG, self).__init__()
        
        self.args = args
        self.kg_name = 'freebase' if 'FB' in args.data_name else 'wikidata'

        self.jsonl_entities = list(open(os.path.join(args.data_path, f'{FOLDER_NAMES[args.data_name]}', 'entities.jsonl'), 'r'))
        self.jsonl_relations = list(open(os.path.join(args.data_path, f'{FOLDER_NAMES[args.data_name]}', 'relations.jsonl'), 'r'))
        self.jsonl_triplets = list(open(os.path.join(args.data_path, f'{FOLDER_NAMES[args.data_name]}', f'{self.kg_name}-kg.jsonl'), 'r'))

        self.entities = self.init_entities(self.jsonl_entities)
        self.relations = self.init_relations(self.jsonl_relations)
        self.triplets = self.init_triplets(self.jsonl_triplets)

    def init_entities(self, jsonl_entities):
        entities = {}
        for entity in jsonl_entities:
            entity = json.loads(entity)
            entities[entity['entity']] = {
                'mention': entity['mention'],
                'aliases': entity['aliases']
            }
        return entities

    def init_relations(self, jsonl_relations):
        relations = {}
        for relation in jsonl_relations:
            relation = json.loads(relation)
            relations[relation['relation']] = {
                'mention': relation['mention']
            }
        return relations

    def init_triplets(self, jsonl_triplets):
        triplets = {}
        for sample in jsonl_triplets:
            sample = json.loads(sample)
            sample['triples'] = [(sample['id'], triplet[0], triplet[1]) for triplet in sample['triples']]
            triplets[sample['id']] = {
                'raw_triplets': sample['triples'],
                'ver_triplets': self.verbalizer(sample['triples'])
            }
        return triplets

    def get_triplets(self, entity_name):
        return self.triplets[entity_name] \
            if entity_name in self.triplets.keys() \
            else {'raw_triplets': [], 'ver_triplets': []}

    def get_aliases(self, entity_name):
        return self.entities[entity_name]['aliases'] \
            if entity_name in self.entities.keys() and self.entities[entity_name]['aliases'] is not None \
            else []

    def verbalizer(self, raw_triplets):
        return [
            [
                self.entities[triplet[0]]['mention'] \
                    if triplet[0] in self.entities.keys()
                    else '',
                self.relations[triplet[1]]['mention'] \
                    if triplet[1] in self.relations.keys() \
                    else '',
                self.entities[triplet[2]]['mention'] \
                    if triplet[2] in self.entities.keys() \
                    else ('' if triplet[2][0] == 'Q' else triplet[2])
            ]
            for triplet in raw_triplets
        ]
