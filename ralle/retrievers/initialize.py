# Copyright (c) Kioxia corporation and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import numpy as np
from .vector_index.initialize import initialize_vector_index

class Retriever:

    def __init__(self, kwargs, query_encoders, corpora):

        self.retriever_type = kwargs['type']
        self.embedding_fn = query_encoders[kwargs['query_encoder_name']] \
                            if 'query_encoder_name' in kwargs else lambda x: x
        self.vector_index = initialize_vector_index(kwargs)[0]
        self.text_fn, self.doc_id_fn, self.doc_id_to_title_fn, \
        self.doc_id_to_corpus_id_fn, self.doc_id_to_wiki_id_fn = corpora[kwargs['corpus_name']]

    def search(self, query, k=1):
        if isinstance(query, str):
            query = [query]
        if self.retriever_type == 'bm25':
            score = []
            doc_id = []
            wiki_id = []
            title = []
            text = []
            for q in query:
                out = self.vector_index.search(q, k)
                score.append([h.score for h in out])
                doc_id.append([int(h.docid) for h in out])
                wiki_id.append([self.doc_id_to_wiki_id_fn(int(h.docid)) for h in out])
                title.append([self.doc_id_to_title_fn(int(h.docid)) for h in out])
                text.append([json.loads(h.raw)['contents'] for h in out])
            return score, doc_id, wiki_id, title, text

        elif self.retriever_type == 'faiss':
            q_emb = self.embedding_fn(query)
            score, out = self.vector_index.search(q_emb, k)
            doc_id = [[self.doc_id_fn(ids) for ids in o] for o in out]
            wiki_id = [[self.doc_id_to_wiki_id_fn(did) for did in d] for d in doc_id]
            title = [[self.doc_id_to_title_fn(ids) for ids in d] for d in doc_id]
            text = [[self.text_fn(ids) for ids in o] for o in out]
            return score, doc_id, wiki_id, title, text

        elif self.retriever_type == 'diskann':
            q_emb = self.embedding_fn(query)
            # print(q_emb.shape)
            w, l = 4, 100

            # print(f"DiskANN first 4 components of query vector: {q_emb.flatten()[:4]}")

            search_count = 0
            search_count_limit = 4

            while True:
                q_res_batch = self.vector_index.batch_search(
                    queries=q_emb,
                    k_neighbors=k, complexity=l, beam_width=w, num_threads=1
                )
                print(len(q_res_batch))
                print(q_res_batch[0].shape)
                print(q_res_batch[1].shape)

                if search_count > 0:
                    score = q_res_batch[1][:-search_count]
                    out = q_res_batch[0][:-search_count]
                else:
                    score = q_res_batch[1]
                    out = q_res_batch[0]

                # print(f"DiskANN search result: {q_res_batch}")
                
                if score[-1, 0] < 1.:
                    break
                
                if search_count > search_count_limit:
                    print(f'DiskANN search failed {search_count_limit} times. Aborting...')
                    break
                
                print('DiskANN search failed. Retrying...')
                search_count += 1
                q_emb = np.concatenate((q_emb, q_emb[-1:]), axis=0)

            doc_id = [[self.doc_id_fn(ids) for ids in o] for o in out]
            wiki_id = [[self.doc_id_to_wiki_id_fn(did) for did in d] for d in doc_id]
            title = [[self.title_fn(ids) for ids in o] for o in out]
            text = [[self.text_fn(ids) for ids in o] for o in out]
            return score, doc_id, wiki_id, title, text

        else:
            raise NotImplementedError
        
    def doc_id_to_corpus(self, doc_id):
        return self.text_fn(self.doc_id_to_corpus_id_fn(doc_id))

def initialize_retriever(kwargs, query_encoders, corpora):
    
    return Retriever(kwargs, query_encoders, corpora)




