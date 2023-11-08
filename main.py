import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import random
import pickle
import logging

from tqdm import tqdm

import numpy as np

import torch

from transformers import HfArgumentParser, set_seed

from configs import GlobalArguments, DataArguments, LanguageModelArguments, RetrieverArguments, VerifierArguments
from data import get_qa_datasets, KG
from models.language_models import LanguageModel
from models.retrievers import Retriever_KG, Retriever_Wiki
from models.verifiers import Verifier_KG, Verifier_Wiki
from metrics import accuracy, f1, em

RETRIEVER_NAMES = {
    'kg': Retriever_KG,
    'wiki': Retriever_Wiki
}

VERIFIER_NAMES = {
    'kg': Verifier_KG,
    'wiki': Verifier_Wiki
}

class Runner(object):
    def __init__(self, glb_args, data_args, lm_args, ret_args, ver_args):
        super(Runner, self).__init__()

        self.glb_args = glb_args
        self.data_args = data_args
        self.lm_args = lm_args
        self.ret_args = ret_args
        self.ver_args = ver_args

        self.output_path = self.get_output_path()
        self.logger = self.get_logger()
        self.logger.info(f"Global Arguments: {glb_args}")
        self.logger.info(f"Dataset Arguments: {data_args}")
        self.logger.info(f"Language Model Arguments: {lm_args}")
        self.logger.info(f"Retriever Arguments: {ret_args}")
        self.logger.info(f"Verifier Arguments: {ver_args}")

        set_seed(glb_args.seed)

        self.qa_datasets, self.qa_data_loaders = get_qa_datasets(data_args)
        self.kg = KG(data_args) if data_args.data_type == 'KGQA' else None
        self.lang_model = LanguageModel(lm_args)
        self.retriever = RETRIEVER_NAMES[glb_args.knowledge_base](ret_args)
        self.verifier = VERIFIER_NAMES[glb_args.knowledge_base](ver_args)
        self.verifier_dataset = None

    def get_logger(self):
        logging.basicConfig(
            format = "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt = "%m/%d/%Y %H:%M:%S",
            level = logging.INFO,
            filename = f"{self.output_path}/logs.txt"
        )
        logger = logging.getLogger(__name__)
        return logger

    def retrieve_generate(self, samples, turn=0):
        questions = [sample['question'] for sample in samples]
        knowledges, knowledges_ids = self.retriever.retrieve(samples, kg=self.kg, offset=turn) \
                                     if self.ret_args.use_retrieval else ([[] for _ in range(len(samples))], [[] for _ in range(len(samples))])
        answers = self.lang_model.generate(questions, knowledges, self.glb_args.knowledge_base)
        return questions, knowledges, knowledges_ids, answers

    def verify(self, questions, knowledges, answers, instruction_sets):
        predictions = [
            self.verifier.verify(questions, knowledges, answers, instruction_index=index) \
            if self.ver_args.use_verification else ([], []) \
            for index in instruction_sets
        ]
        
        pred_all_probs = np.array([prediction[1] for prediction in predictions])[:, :3, :] if self.ver_args.use_verification else []
        pred_probs = np.sum(pred_all_probs, axis=0) if self.ver_args.use_verification else []
        pred_labels = np.argmax(pred_probs, 0) if self.ver_args.use_verification else []
        return pred_all_probs, pred_probs, pred_labels

    def edit(self, samples, questions, knowledges, knowledges_ids, answers, instruction_sets, pred_labels, verifier_labels, turn_id):
        ret_indices, gen_indices, cor_indices = \
            np.where(pred_labels == 0)[0], np.where(pred_labels == 1)[0], np.where(pred_labels == 2)[0]
        
        # Retrieval
        ret_samples = [samples[index] for index in ret_indices]
        _, ret_knowledges, ret_knowledges_ids, ret_answers = self.retrieve_generate(ret_samples, turn=turn_id) \
                                                             if len(ret_samples) != 0 else ([], [], [], [])

        # Generation
        gen_questions = [questions[index] for index in gen_indices]
        gen_knowledges = [knowledges[index] for index in gen_indices]
        get_answers = self.lang_model.generate(gen_questions, gen_knowledges, self.glb_args.knowledge_base) \
                      if len(gen_questions) != 0 else []

        # Merge
        knowledges, knowledges_ids = knowledges[:], knowledges_ids[:]
        for index, ret_index in enumerate(ret_indices):
            knowledges[ret_index], knowledges_ids[ret_index] = ret_knowledges[index], ret_knowledges_ids[index]

        answers = np.array(answers)
        answers[ret_indices] = ret_answers
        answers[gen_indices] = get_answers
        answers = answers.tolist()

        # Verification
        verifier_labels = self.verifier.get_labels(samples, knowledges, knowledges_ids, answers, self.kg, self.data_args.aliases, instruction_index=0)
        pred_all_probs, pred_probs, pred_labels = self.verify(questions, knowledges, answers, instruction_sets)
        return answers, pred_all_probs, pred_probs, pred_labels, verifier_labels

    def edit_loop(self, samples, questions, knowledges, knowledges_ids, answers, instruction_sets, pred_labels, verifier_labels):
        for turn_id in range(1, self.glb_args.num_edits+1):
            answers, pred_all_probs, pred_probs, pred_labels, verifier_labels = \
                self.edit(samples, questions, knowledges, knowledges_ids, answers, instruction_sets, pred_labels, verifier_labels, turn_id)
        return answers, pred_all_probs, pred_probs, pred_labels, verifier_labels

    def eval(self):
        results = {
            'llm_answers': [],
            'ver_true_labels': [],
            'ver_pred_labels': [],
            'ver_pred_probs': [],
            'ver_pred_all_probs': []
        }

        for index, batch in enumerate(tqdm(self.qa_data_loaders['test'])):
            if self.glb_args.debug and index == 30:  break
            
            questions, knowledges, knowledges_ids, answers = self.retrieve_generate(batch)

            instruction_sets = [i for i in range(self.ver_args.verifier_num_instructions)] if self.ver_args.ensemble else [0]
            verifier_labels = self.verifier.get_labels(batch, knowledges, knowledges_ids, answers, self.kg, self.data_args.aliases, instruction_index=0) \
                              if self.ver_args.use_verification else []

            pred_all_probs, pred_probs, pred_labels = self.verify(questions, knowledges, answers, instruction_sets)

            if self.glb_args.edit_output == True:
                answers, pred_all_probs, pred_probs, pred_labels, verifier_labels = \
                    self.edit_loop(batch, questions, knowledges, knowledges_ids, answers, instruction_sets, pred_labels, verifier_labels)

            results['llm_answers'].extend(answers)
            results['ver_true_labels'].extend(verifier_labels)
            results['ver_pred_labels'].extend(pred_labels)
            results['ver_pred_probs'].append(pred_probs)
            results['ver_pred_all_probs'].append(pred_all_probs)
        results['ver_pred_probs'] = np.concatenate(results['ver_pred_probs'], axis=1) if self.ver_args.use_verification else []

        return {
            'samples': self.qa_datasets['test'],
            'llm_answers': results['llm_answers'],
            'generation_accuracy': accuracy(self.qa_datasets['test'], results['llm_answers'], self.kg, data_args.aliases),
            'generation_f1': f1(self.qa_datasets['test'], results['llm_answers'], self.kg, data_args.aliases),
            'generation_em': em(self.qa_datasets['test'], results['llm_answers'], self.kg, data_args.aliases),
            'verification_true_labels': results['ver_true_labels'],
            'verification_pred_labels': results['ver_pred_labels'],
            'verification_pred_probs': results['ver_pred_probs'],
            'verification_accuracy': self.verifier.measure_perf(results['ver_true_labels'], results['ver_pred_labels'])
        }

    def get_verifier_dataset(self):
        verifier_dataset = {
            'text': [],
            'labels': []
        }

        for index, batch in enumerate(tqdm(self.qa_data_loaders['train'])):
            if self.glb_args.debug and index == 30:  break
            if self.ver_args.verifier_sample and index == 1000: break

            questions, knowledges, knowledges_ids, answers = self.retrieve_generate(batch)

            instruction_index = random.randint(0, self.ver_args.verifier_num_instructions-1) if self.ver_args.ensemble else 0
            verifier_inputs = self.verifier.get_prompts(questions, knowledges, answers, instruction_index)
            verifier_labels = self.verifier.get_labels(batch, knowledges, knowledges_ids, answers, self.kg, self.data_args.aliases, instruction_index)
            
            verifier_dataset['text'].extend(verifier_inputs)
            verifier_dataset['labels'].extend([self.verifier.label_to_option(label) for label in verifier_labels])

        self.verifier_dataset = verifier_dataset
        return verifier_dataset

    def train_verifier(self):
        verifier_dataset = self.get_verifier_dataset() if self.verifier_dataset is None else self.verifier_dataset
        loss = self.verifier.update(verifier_dataset)
        return loss

    def run(self):
        best_index, all_results = 0, []

        for epoch in tqdm(range(self.ver_args.verifier_num_epochs)):
            if not self.ver_args.use_verification and epoch == 1:
                break

            if self.ver_args.use_verification:
                loss = self.train_verifier()
                self.logger.info(f"[EPOCH: {epoch}] Averaged Training Loss: {loss}")

            with torch.no_grad():
                all_results.append(self.eval())
                self.logger.info(f"[EPOCH: {epoch}] Verification Accuracy: {all_results[-1]['verification_accuracy']}")

            if all_results[best_index]['verification_accuracy'] <= all_results[-1]['verification_accuracy']:
                best_index = epoch

        self.save_results(all_results[best_index], all_results)

    def get_output_path(self):
        output_path = f"./results/{self.data_args.data_name}/{self.lm_args.model_name_or_path.split('/')[-1]}/{self.glb_args.exp_name}"
        if not os.path.exists(output_path): os.makedirs(output_path)
        print(f"OUTPUT_PATH: {output_path}")
        return output_path

    def save_results(self, best_results, all_results):
        with open(f"{self.output_path}/all_results.pkl", "wb") as outfile:
            pickle.dump(all_results, outfile)
        with open(f"{self.output_path}/best_results.pkl", "wb") as outfile:
            pickle.dump(best_results, outfile)

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

    runner = Runner(glb_args, data_args, lm_args, ret_args, ver_args)
    runner.run()
    if glb_args.stop:   import pdb; pdb.set_trace()
