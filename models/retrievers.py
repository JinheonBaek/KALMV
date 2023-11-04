import os
import itertools

import numpy as np

from sentence_transformers import SentenceTransformer, util
from pyserini.search.lucene import LuceneSearcher

MODEL_NAMES = {
    "mpnet": "sentence-transformers/all-mpnet-base-v2"
}

class Retriever_KG(object):
    def __init__(self, args):
        super(Retriever_KG, self).__init__()

        self.args = args

        self.model = self.get_model()

    def get_model(self):
        return SentenceTransformer(
            MODEL_NAMES[self.args.retriever_name],
            cache_folder = self.args.cache_dir
        )

    def get_triplets(self, samples, kg):
        batch_question_entities = [list(set(
            [entity['name'] for entity in sample['question_entities']])) 
            for sample in samples
        ]
        batch_raw_triplets = [list(itertools.chain(*
            [kg.get_triplets(entity)['raw_triplets'] for entity in question_entities]))
            for question_entities in batch_question_entities
        ]
        batch_ver_triplets = [list(itertools.chain(*
            [kg.get_triplets(entity)['ver_triplets'] for entity in question_entities]))
            for question_entities in batch_question_entities
        ]

        assert len(samples) == \
               len(batch_question_entities) == \
               len(batch_raw_triplets) == \
               len(batch_ver_triplets)
        
        return batch_raw_triplets, batch_ver_triplets

    def encode(self, sentences):
        return self.model.encode(
            sentences,
            batch_size=self.args.retriever_batch_size,
            show_progress_bar=False,
        )

    def filter_triplets(self, raw_triplets, ver_triplets):
        indices = [
            index for index, triplet in enumerate(ver_triplets)
            if triplet[0] not in [None, ''] and \
               triplet[1] not in [None, ''] and \
               triplet[2] not in [None, '']
        ]
        return [raw_triplets[index] for index in indices], \
               [ver_triplets[index] for index in indices]

    def retrieve(self, samples, kg, return_ids=True, offset=0, *args, **kwargs):
        batch_raw_triplets, batch_ver_triplets = self.get_triplets(samples, kg)

        raw_results, ver_results = [], []

        for sample, raw_triplets, ver_triplets in zip(samples, batch_raw_triplets, batch_ver_triplets):
            assert len(raw_triplets) == len(ver_triplets)

            query = sample['question']
            raw_triplets, ver_triplets = self.filter_triplets(raw_triplets, ver_triplets)

            if len(ver_triplets) == 0:
                raw_results.append([])
                ver_results.append([])
                continue
            
            query_embedding = self.encode([query])
            triplet_embeddings = self.encode([
                f"{triplet[0]}{self.args.retriever_sep_token}{triplet[1]}{self.args.retriever_sep_token}{triplet[2]}" 
                for triplet in ver_triplets
            ])
            similarities = util.dot_score(query_embedding, triplet_embeddings)[0].numpy()

            sorted_indices = np.argsort(-similarities)
            sorted_raw_triples = [raw_triplets[index] for index in sorted_indices]
            sorted_ver_triples = [ver_triplets[index] for index in sorted_indices]

            raw_results.append(sorted_raw_triples[offset*self.args.retriever_top_k:(offset+1)*self.args.retriever_top_k])
            ver_results.append(sorted_ver_triples[offset*self.args.retriever_top_k:(offset+1)*self.args.retriever_top_k])

        return ver_results if return_ids == False \
               else (ver_results, raw_results)
    
class Retriever_Wiki(object):
    def __init__(self, args):
        super(Retriever_Wiki, self).__init__()

        self.args = args

        self.set_cache_dir(self.args.index_dir)
        self.model = self.get_model()

    def set_cache_dir(self, dir):
        os.environ['PYSERINI_CACHE'] = dir

    def get_model(self):
        return LuceneSearcher.from_prebuilt_index('enwiki-paragraphs')
    
    def retrieve(self, samples, return_ids=True, offset=0, *args, **kwargs):
        results = [
            self.model.search(sample['question'])[offset:]
            for sample in samples
        ]

        raw_results = [
            [(doc.docid, doc.score) for doc in docs] 
            for docs in results
        ]

        ver_results = [
            [doc.raw for doc in docs]
            for docs in results
        ]

        return ver_results if return_ids == False \
               else (ver_results, raw_results)