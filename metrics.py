import re
import string
import itertools
import collections

import numpy as np

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()

def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))

def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def em(samples, pred_answers, kg, aliases=True):
    if not isinstance(pred_answers, list):
        samples = [samples]
        pred_answers = [pred_answers]

    assert len(samples) == len(pred_answers)

    num_all_answers = 0
    num_correct_answers = 0
    for sample, pred_answer in zip(samples, pred_answers):
        gold_entities = set(list(itertools.chain(*
            [[entity['mention'] for entity in sample['answer_entities'] if entity['mention'] not in [None, '']]] + \
            [kg.get_aliases(entity['name']) for entity in sample['answer_entities'] if aliases == True and kg is not None and entity['name'] is not None]
        )))
        if len(gold_entities) == 0: continue

        num_all_answers += 1
        num_correct_answers += 1 \
            if np.count_nonzero([compute_exact(gold_entity, pred_answer) for gold_entity in gold_entities]) != 0 \
            else 0
        
    return num_correct_answers / (num_all_answers + 1e-16)

def f1(samples, pred_answers, kg, aliases=True):
    if not isinstance(pred_answers, list):
        samples = [samples]
        pred_answers = [pred_answers]

    assert len(samples) == len(pred_answers)

    num_all_answers = 0
    num_correct_answers = 0
    for sample, pred_answer in zip(samples, pred_answers):
        gold_entities = set(list(itertools.chain(*
            [[entity['mention'] for entity in sample['answer_entities'] if entity['mention'] not in [None, '']]] + \
            [kg.get_aliases(entity['name']) for entity in sample['answer_entities'] if aliases == True and kg is not None and entity['name'] is not None]
        )))
        if len(gold_entities) == 0: continue

        num_all_answers += 1
        num_correct_answers += max([compute_f1(gold_entity, pred_answer) for gold_entity in gold_entities])
        
    return num_correct_answers / (num_all_answers + 1e-16)

def accuracy(samples, pred_answers, kg, aliases=True):
    if not isinstance(pred_answers, list):
        samples = [samples]
        pred_answers = [pred_answers]

    assert len(samples) == len(pred_answers)

    num_all_answers = 0
    num_correct_answers = 0
    for sample, pred_answer in zip(samples, pred_answers):
        gold_entities = set(list(itertools.chain(*
            [[entity['mention'] for entity in sample['answer_entities'] if entity['mention'] not in [None, '']]] + \
            [kg.get_aliases(entity['name']) for entity in sample['answer_entities'] if aliases == True and kg is not None and entity['name'] is not None]
        )))
        if len(gold_entities) == 0: continue

        num_all_answers += 1
        num_correct_answers += 1 \
            if np.count_nonzero([normalize_answer(gold_entity) in normalize_answer(pred_answer) for gold_entity in gold_entities]) != 0 \
            else 0
        
    return num_correct_answers / (num_all_answers + 1e-16)