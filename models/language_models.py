import torch
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)

class LanguageModel(object):
    def __init__(self, args):
        super(LanguageModel, self).__init__()

        self.args = args

        self.config = self.get_config()
        self.tokenizer = self.get_tokenizer()
        self.model = self.get_model()

    def get_config(self):
        config = AutoConfig.from_pretrained(
            self.args.model_name_or_path,
            cache_dir = self.args.cache_dir,
        )
        return config

    def get_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(
            self.args.model_name_or_path,
            cache_dir = self.args.cache_dir,
            resume_download = True
        )
        return tokenizer

    def get_model(self):
        model = AutoModelForSeq2SeqLM.from_pretrained(
            self.args.model_name_or_path,
            config = self.config,
            cache_dir = self.args.cache_dir,
            torch_dtype = torch.float16 if self.args.fp16 else None,
            device_map = self.args.device_map if self.args.n_gpu >= 0 else None,
            resume_download = True
        )
        return model
    
    def verbalize_knowledge(self, knowledge, knowledge_base):
        if not knowledge:
            return ''

        if knowledge_base == 'wiki':
            return f'Below is the passage meaningful to answer the question. \n{knowledge[0]} \n\n'

        string = 'Below are facts in the form of the triple meaningful to answer the question. \n'
        for triplet in knowledge:
            string += f'({triplet[0]}, {triplet[1]}, {triplet[2]}) \n'
        string += '\n'
        return string

    def generate(self, questions, knowledges, knowledge_base='kg'):
        prompts = [
            f'{self.verbalize_knowledge(knowledge, knowledge_base)}{self.args.question_prefix}{question}\n{self.args.question_postfix}' \
            for (question, knowledge) in zip(questions, knowledges)
        ]

        input_tokens = self.tokenizer(
            prompts, max_length=self.args.max_source_length, padding='longest', truncation=True, return_tensors='pt',
        ).to(self.args.device)

        generated_tokens = self.model.generate(
            **input_tokens,
            max_new_tokens=self.args.max_target_length,
            do_sample=True,
            top_k=40,
            temperature=0.5,
        )

        outputs = self.tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True
        )
        return outputs