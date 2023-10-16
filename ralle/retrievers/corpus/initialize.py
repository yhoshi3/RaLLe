# Copyright (c) Kioxia corporation and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pickle
import linecache
import subprocess
from time import time

def initialize_corpus(kwargs, skip_load_corpus=False):

    t0 = time(); print('loading doc_id to wikipedia_id mapping...')
    with open(kwargs['doc_id_to_wikipedia_id_mapping_path'], 'rb') as f:
        doc_id_to_wikipedia_id_map = pickle.load(f)
    t1 = time(); print('loading doc_id to wikipedia_id mapping...done ({:.2f} sec)'.format(t1-t0))

    def doc_id_to_wikipedia_id_mapping_fn(ids):
        return doc_id_to_wikipedia_id_map[ids]

    if skip_load_corpus:
        def corpus_text_fn(ids):
            return None

        def corpus_doc_id_fn(ids):
            return None
        
        def doc_id_to_title_fn(ids):
            return None

        def doc_id_to_corpus_id_fn(doc_id):
            return None

    else:
        t0 = time(); print('loading corpus...')
        def corpus_text_fn(ids):
            text = linecache.getline(kwargs['corpus_path'], ids + 1)
            return text.split(kwargs['delimiter'])[kwargs['col_text']].strip()

        def corpus_doc_id_fn(ids):
            text = linecache.getline(kwargs['corpus_path'], ids + 1)
            return int(text.split(kwargs['delimiter'])[kwargs['col_doc_id']].strip())

        def doc_id_to_title_fn(ids):
            text = linecache.getline(kwargs['corpus_path'], ids)
            return text.split(kwargs['delimiter'])[kwargs['col_title']].strip()

        num_lines = int(subprocess.check_output(['wc', '-l', kwargs['corpus_path']]).decode().split(' ')[0])
        doc_id_to_corpus_id = {corpus_doc_id_fn(i): i for i in range(num_lines)}
        def doc_id_to_corpus_id_fn(doc_id):
            return doc_id_to_corpus_id[doc_id]
        t1 = time(); print('loading corpus...done ({:.2f} sec)'.format(t1-t0))

    return corpus_text_fn, corpus_doc_id_fn, doc_id_to_title_fn, doc_id_to_corpus_id_fn, doc_id_to_wikipedia_id_mapping_fn
