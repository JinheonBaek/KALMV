import numpy as np

from tqdm import tqdm

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq
)
from datasets import Dataset

from metrics import normalize_answer, accuracy, f1, em

METRIC_NAMES = {
    'accuracy': accuracy,
    'f1': f1,
    'em': em
}

INSTRUCTIONS_KG = [
"""The following is a multiple choice question about a question answering task. In this task, the AI model generates an output given a question with facts represented in the triplet form. The facts are retrieved from knowledge graphs, which may or may not be helpful to answer the question.

Question: {question}
Facts: {facts}
Output: {answer}

Options:
A. Facts are unhelpful to answer the question.
B. Facts are helpful to answer the question, yet the generated output for the question is incorrect.
C. The generated output for the question is correct.

Select one option:""",

"""Question: {question}
Facts: {facts}
Output: {answer}

Options:
A. Facts are unhelpful to answer the question.
B. Facts are helpful to answer the question, yet the generated output for the question is incorrect.
C. The generated output for the question is correct.

Select one option:""",

"""Given a question and some facts from knowledge graphs, the AI model generates an output as follows: 

Question: {question}
Facts: {facts}
Output: {answer}

This is a multiple choice question, and, based on the above information, you need to select one option among three, as follows:

A. Facts are unhelpful to answer the question.
B. Facts are helpful to answer the question, yet the generated output for the question is incorrect.
C. The generated output for the question is correct.

Select one option:""",

"""Here is a question, facts, and generated output from the question and facts. Based on them, you need to select one option among the three.

Question: {question}
Facts: {facts}
Output: {answer}

Options:
A. Facts are unhelpful to answer the question.
B. Facts are helpful to answer the question, yet the generated output for the question is incorrect.
C. The generated output for the question is correct.

Select one option:""",

"""Given a question, facts, and output, which option is the best?

Question: {question}
Facts: {facts}
Output: {answer}

Options:
A. Facts are unhelpful to answer the question.
B. Facts are helpful to answer the question, yet the generated output for the question is incorrect.
C. The generated output for the question is correct.

Select one option:"""
]

INSTRUCTIONS_Wiki = [
"""The following is a multiple choice question about a question answering task. In this task, the AI model generates an output given a question with a passage. The passage is retrieved from Wikipedia, which may or may not be helpful to answer the question.

Question: {question}
Passage: {passage}
Output: {answer}

Options:
A. The passage is unhelpful to answer the question.
B. The passage is helpful to answer the question, yet the generated output for the question is incorrect.
C. The generated output for the question is correct.

Select one option:""",

"""Question: {question}
Passage: {passage}
Output: {answer}

Options:
A. The passage is unhelpful to answer the question.
B. The passage is helpful to answer the question, yet the generated output for the question is incorrect.
C. The generated output for the question is correct.

Select one option:""",

"""Given a question and a passage from Wikipedia, the AI model generates an output as follows: 

Question: {question}
Passage: {passage}
Output: {answer}

This is a multiple choice question, and, based on the above information, you need to select one option among three, as follows:

A. The passage is unhelpful to answer the question.
B. The passage is helpful to answer the question, yet the generated output for the question is incorrect.
C. The generated output for the question is correct.

Select one option:""",

"""Here is a question, passage, and generated output from the question and passage. Based on them, you need to select one option among the three.

Question: {question}
Passage: {passage}
Output: {answer}

Options:
A. The passage is unhelpful to answer the question.
B. The passage is helpful to answer the question, yet the generated output for the question is incorrect.
C. The generated output for the question is correct.

Select one option:""",

"""Given a question, passage, and output, which option is the best?

Question: {question}
Passage: {passage}
Output: {answer}

Options:
A. The passage is unhelpful to answer the question.
B. The passage is helpful to answer the question, yet the generated output for the question is incorrect.
C. The generated output for the question is correct.

Select one option:"""
]


class Verifier(object):
    def __init__(self, args):
        super(Verifier, self).__init__()

        self.args = args
        self.args.max_source_length = args.verifier_max_source_length
        self.args.max_target_length = args.verifier_max_target_length

        self.generation_metric = METRIC_NAMES[args.verifier_generation_metric]
        self.generation_threshold = args.verifier_generation_threshold
        self.retrieval_threshold = args.verifier_retrieval_threshold

        self.config = self.get_config()
        self.tokenizer = self.get_tokenizer()
        self.model = self.get_model()
        self.optimizer = self.get_optimizer()
        self.data_collator = self.get_data_collator()

    def get_config(self):
        config = AutoConfig.from_pretrained(
            self.args.verifier_name,
            cache_dir = self.args.cache_dir,
        )
        return config

    def get_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(
            self.args.verifier_name,
            cache_dir = self.args.cache_dir,
            resume_download = True
        )
        return tokenizer

    def get_model(self):
        model = AutoModelForSeq2SeqLM.from_pretrained(
            self.args.verifier_name,
            config = self.config,
            cache_dir = self.args.cache_dir,
            device_map = self.args.device_map if self.args.n_gpu >= 0 else None,
            resume_download = True
        )
        return model

    def get_optimizer(self):
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.verifier_weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr = self.args.verifier_learning_rate)
        return optimizer

    def get_data_collator(self):
        data_collator = DataCollatorForSeq2Seq(
            self.tokenizer,
            model = self.model,
            label_pad_token_id = -100,
            pad_to_multiple_of = None # For FP16
        )
        return data_collator

    def get_prompts(self, questions, knowledges, answers, instruction_index=0):
        raise NotImplementedError()

    def get_labels(self, samples, knowledges_ids, pred_answers, kg, aliases=True, instruction_index=0):
        raise NotImplementedError()

    def option_to_label(self, option):
        return {
            'A': 0,
            'B': 1,
            'C': 2,
        }[option]

    def label_to_option(self, label):
        return {
            0: 'A',
            1: 'B',
            2: 'C',
        }[label]

    def preprocess_function(self, examples):
        model_inputs = self.tokenizer(examples['text'], max_length=self.args.max_source_length, padding='longest', truncation=True, return_tensors='pt')
        labels = self.tokenizer(text_target=examples['labels'], max_length=self.args.max_target_length+1, padding='longest', truncation=True, return_tensors='pt')['input_ids']
        labels[labels == self.tokenizer.pad_token_id] = -100
        model_inputs['labels'] = labels
        return model_inputs

    def update(self, dataset):
        dataset = Dataset.from_dict(dataset).map(self.preprocess_function, batched=True).remove_columns(['text'])
        dataset.set_format('torch')
        data_loader = DataLoader(dataset, shuffle=True, batch_size=self.args.verifier_batch_size, collate_fn=self.data_collator)

        self.model.train()

        total_loss = 0
        for batch in data_loader:
            outputs = self.model(
                input_ids=batch['input_ids'].to(self.args.device),
                attention_mask=batch['attention_mask'].to(self.args.device),
                labels=batch['labels'].to(self.args.device)
            )
            loss = outputs.loss
            loss.backward()
            total_loss += loss.cpu().detach().float()

            self.optimizer.step()
            self.optimizer.zero_grad()

        self.model.eval()
        return total_loss / len(data_loader)

    def verify(self, questions, knowledges, answers, instruction_index, return_probs=True):
        prompts = self.get_prompts(questions, knowledges, answers, instruction_index)

        input_tokens = self.tokenizer(
            prompts, max_length=self.args.max_source_length, padding='longest', truncation=True, return_tensors='pt',
        ).to(self.args.device)
        
        scores = self.model.generate(
            **input_tokens,
            max_new_tokens=self.args.max_target_length,
            return_dict_in_generate=True,
            output_scores=True
        ).scores[0]

        probs = (
            torch.nn.functional.softmax(
                torch.stack([
                    scores[:, self.tokenizer('A').input_ids[0]],
                    scores[:, self.tokenizer('B').input_ids[0]],
                    scores[:, self.tokenizer('C').input_ids[0]],
                ]), dim=0,
            ).detach().cpu().numpy()
        )
        preds = np.argmax(probs, 0)

        return preds if return_probs == False \
               else (preds, probs)

    def measure_perf(self, gold_labels, pred_labels):
        return sum([1 for (x, y) in zip(gold_labels, pred_labels) if x == y]) / \
               (len(gold_labels) + 1e-16)


class Verifier_KG(Verifier):
    def __init__(self, args):
        super(Verifier_KG, self).__init__(args)

        self.args.instructions = INSTRUCTIONS_KG

    def verbalize_knowledge(self, knowledge):
        string = ''
        for triplet in knowledge:
            string += f'({triplet[0]}, {triplet[1]}, {triplet[2]}){self.args.fact_sep_token}'
        return string

    def get_prompts(self, questions, knowledges, answers, instruction_index=0):
        if not isinstance(questions, list):
            questions = [questions]
            knowledges = [knowledges]
            answers = [answers]

        return [
            self.args.instructions[instruction_index].format(
                question = question,
                facts = self.verbalize_knowledge(knowledge),
                answer = answer
            ) for (question, knowledge, answer) in zip(questions, knowledges, answers)
        ]

    def get_labels(self, samples, knowledges, knowledges_ids, pred_answers, kg, aliases=True, instruction_index=0):
        labels = np.zeros(len(samples), dtype=int)

        knowledges_entity_ids = [list(set(
            [triplet[0] for triplet in knowledge_ids if triplet[0].startswith('Q')] + \
            [triplet[2] for triplet in knowledge_ids if triplet[2].startswith('Q')])) 
            for knowledge_ids in knowledges_ids
        ]
        answers_entity_ids = [list(set(
            [entity['name'] for entity in sample['answer_entities'] 
             if entity['name'] is not None and entity['mention'] is not None]))
            for sample in samples
        ]
        retrieval_scores = np.array([
            len(set(knowledge_entities) & set(answer_entities)) / (len(answer_entities) + 1e-16)
            for (knowledge_entities, answer_entities) 
            in zip(knowledges_entity_ids, answers_entity_ids)
        ])
        labels[np.where(retrieval_scores > self.retrieval_threshold)[0]] = 1

        generation_scores = np.array([
            self.generation_metric(sample, pred_answer, kg, aliases) \
            for (sample, pred_answer) in zip(samples, pred_answers)
        ])
        labels[np.where(generation_scores > self.generation_threshold)[0]] = 2

        return labels


class Verifier_Wiki(Verifier):
    def __init__(self, args):
        super(Verifier_Wiki, self).__init__(args)

        self.args.instructions = INSTRUCTIONS_Wiki

    def get_prompts(self, questions, knowledges, answers, instruction_index=0):
        if not isinstance(questions, list):
            questions = [questions]
            knowledges = [knowledges]
            answers = [answers]

        return [
            self.args.instructions[instruction_index].format(
                question = question,
                passage = knowledge[0],
                answer = answer
            ) for (question, knowledge, answer) in zip(questions, knowledges, answers)
        ]

    def get_labels(self, samples, knowledges, knowledges_ids, pred_answers, kg, aliases=True, instruction_index=0):
        labels = np.zeros(len(samples), dtype=int)

        answers_entities = [list(set(
            [entity['mention'] for entity in sample['answer_entities'] 
             if entity['name'] is not None and entity['mention'] is not None]))
            for sample in samples
        ]
        retrieval_scores = np.array([
            np.count_nonzero([normalize_answer(entity) in normalize_answer(knowledge[0]) for entity in answer_entities]) / (len(answer_entities) + 1e-16)
            for (knowledge, answer_entities)
            in zip(knowledges, answers_entities)
        ])
        labels[np.where(retrieval_scores > self.retrieval_threshold)[0]] = 1

        generation_scores = np.array([
            self.generation_metric(sample, pred_answer, kg, aliases) \
            for (sample, pred_answer) in zip(samples, pred_answers)
        ])
        labels[np.where(generation_scores > self.generation_threshold)[0]] = 2

        return labels
