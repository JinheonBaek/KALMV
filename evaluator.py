import pickle
import logging

import numpy as np

import torch

from transformers import HfArgumentParser

from main import Runner
from configs import GlobalArguments, DataArguments, LanguageModelArguments, RetrieverArguments, VerifierArguments
from metrics import accuracy, f1, em

METRIC_NAMES = {
    'acc': accuracy,
    'f1': f1,
    'em': em
}

class Evaluator(Runner):
    def __init__(self, glb_args, data_args, lm_args, ret_args, ver_args):
        super(Evaluator, self).__init__(glb_args, data_args, lm_args, ret_args, ver_args)

        self.results = self.get_results()

    def get_logger(self):
        logging.basicConfig(
            format = "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt = "%m/%d/%Y %H:%M:%S",
            level = logging.INFO,
            filename = f"{self.output_path}/logs_eval.txt"
        )
        logger = logging.getLogger(__name__)
        return logger

    def get_results(self):
        with open(f"{self.output_path}/best_results.pkl", "rb") as f:
            results = pickle.load(f)
        return results

    def get_answer_score(self, metric='acc', is_filtered=False):
        metric = METRIC_NAMES[metric]

        target_indices = [
            index for index, pred_label in enumerate(self.results['verification_pred_labels']) \
            if pred_label == 2
        ] if is_filtered else [index for index in range(len(self.results['verification_pred_labels']))]
        
        return metric(
            [self.qa_datasets['test'][index] for index in target_indices],
            [self.results['llm_answers'][index] for index in target_indices],
            self.kg, data_args.aliases
        )

    def get_verifier_precision_recall_f1(self, target_label=2):
        predicted_correct_indices = set([
            index for index, pred_label in enumerate(self.results['verification_pred_labels']) \
            if pred_label == target_label
        ])
        actual_correct_indices = set([
            index for index, true_label in enumerate(self.results['verification_true_labels']) \
            if true_label == target_label
        ])

        precision = len(predicted_correct_indices & actual_correct_indices) / len(predicted_correct_indices)
        recall = len(predicted_correct_indices & actual_correct_indices) / len(actual_correct_indices)
        f1 = 2 * (precision * recall) / (precision + recall)
        return precision, recall, f1

    def get_verifier_accuracy(self):
        gold_labels, pred_labels = self.results['verification_true_labels'], self.results['verification_pred_labels']
        
        all_acc = sum([1 for (x, y) in zip(gold_labels, pred_labels) if x == y]) / \
                  (len(gold_labels) + 1e-16)
        ret_acc = sum([1 for (x, y) in zip(gold_labels, pred_labels) if (x == 0 and y == 0) or (x != 0 and y != 0)]) / \
                  (np.sum(np.array(gold_labels) == 0) + np.sum(np.array(gold_labels) != 0) + 1e-16)
        gro_acc = sum([1 for (x, y) in zip(gold_labels, pred_labels) if (x == 1 and y == 1) or (x != 1 and y != 1)]) / \
                  (np.sum(np.array(gold_labels) == 1) + np.sum(np.array(gold_labels) != 1) + 1e-16)
        gen_acc = sum([1 for (x, y) in zip(gold_labels, pred_labels) if (x == 2 and y == 2) or (x != 2 and y != 2)]) / \
                  (np.sum(np.array(gold_labels) == 2) + np.sum(np.array(gold_labels) != 2) + 1e-16)
        return all_acc, ret_acc, gro_acc, gen_acc

    def get_label_statistics(self):
        gold_labels, pred_labels = self.results['verification_true_labels'], self.results['verification_pred_labels']

        return {
            'gold_retrieval_count': np.sum(np.array(gold_labels) == 0),
            'gold_grounding_count': np.sum(np.array(gold_labels) == 1),
            'gold_generation_count': np.sum(np.array(gold_labels) == 2),
            'pred_retrieval_count': np.sum(np.array(pred_labels) == 0),
            'pred_grounding_count': np.sum(np.array(pred_labels) == 1),
            'pred_generation_count': np.sum(np.array(pred_labels) == 2),
        }

    def eval(self):
        original_answer_acc, original_answer_f1, original_answer_em = \
            self.get_answer_score('acc'), self.get_answer_score('f1'), self.get_answer_score('em')
        filtered_answer_acc, filtered_answer_f1, filtered_answer_em = \
            self.get_answer_score('acc', is_filtered=True), self.get_answer_score('f1', is_filtered=True), self.get_answer_score('em', is_filtered=True)
        verifier_precision, verifier_recall, verifier_f1 = self.get_verifier_precision_recall_f1(target_label=2)
        all_acc, retrieval_acc, grounding_acc, generation_acc = self.get_verifier_accuracy()
        label_stats = self.get_label_statistics()

        self.logger.info(f"[Answer Generation Results] Original - Acc: {original_answer_acc}, F1: {original_answer_f1}, EM: {original_answer_em}")
        self.logger.info(f"[Answer Generation Results] Filtered - Acc: {filtered_answer_acc}, F1: {filtered_answer_f1}, EM: {filtered_answer_em}")
        self.logger.info(f"[Verification Results] Precision: {verifier_precision}, Recall: {verifier_recall}, F1: {verifier_f1}")
        self.logger.info(f"[Verification Results] All Acc: {all_acc}, Retrieval Acc: {retrieval_acc}, Grounding Acc: {grounding_acc}, Generation Acc: {generation_acc}")
        self.logger.info(f"[Label Statistics] Gold - Retrieval Num: {label_stats['gold_retrieval_count']}, Grounding Num: {label_stats['gold_grounding_count']}, Generation Num: {label_stats['gold_generation_count']}")
        self.logger.info(f"[Label Statistics] Pred - Retrieval Num: {label_stats['pred_retrieval_count']}, Grounding Num: {label_stats['pred_grounding_count']}, Generation Num: {label_stats['pred_generation_count']}")
        self.summary({
            'generation_accuracy': filtered_answer_acc,
            'generation_f1': filtered_answer_f1,
            'generation_em': filtered_answer_em,
            'verification_accuracy': all_acc
        })

    def summary(self, results):
        with open(f"{self.output_path}/summary.txt", "a") as outfile:
            outfile.write("---" * 10 + "\n")
            outfile.write(f"Global Arguments: {glb_args} \n")
            outfile.write(f"Dataset Arguments: {data_args} \n")
            outfile.write(f"Language Model Arguments: {lm_args} \n")
            outfile.write(f"Retriever Arguments: {ret_args} \n")
            outfile.write(f"Verifier Arguments: {ver_args} \n")
            outfile.write(f"Generation Accuracy: {results['generation_accuracy']} \n")
            outfile.write(f"Generation F1: {results['generation_f1']} \n")
            outfile.write(f"Generation EM: {results['generation_em']} \n")
            outfile.write(f"Verification Accuracy: {results['verification_accuracy']} \n")
            outfile.write("---" * 10 + "\n")

if __name__ == "__main__":

    parser = HfArgumentParser((GlobalArguments, DataArguments, LanguageModelArguments, RetrieverArguments, VerifierArguments))

    glb_args, data_args, lm_args, ret_args, ver_args = parser.parse_args_into_dataclasses()

    glb_args.device = lm_args.device = ret_args.device = ver_args.device = torch.device("cuda" if torch.cuda.is_available() and not glb_args.no_cuda else "cpu")
    glb_args.n_gpu = lm_args.n_gpu = ret_args.n_gpu = ver_args.n_gpu = 0 if glb_args.no_cuda else torch.cuda.device_count()
    ver_args.cache_dir = ret_args.cache_dir = lm_args.cache_dir
    ver_args.device_map = ret_args.device_map = lm_args.device_map

    data_args.data_splits = ['train', 'dev', 'test'] \
        if data_args.data_name in ['Mintaka'] \
        else ['train', 'test']

    evaluator = Evaluator(glb_args, data_args, lm_args, ret_args, ver_args)
    evaluator.eval()
