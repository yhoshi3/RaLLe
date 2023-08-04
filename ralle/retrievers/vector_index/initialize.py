# Copyright (c) Kioxia corporation and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import faiss
import numpy
import os
from time import time

def initialize_bm25(kwargs):

    try:
        from pyserini.search.lucene import LuceneSearcher
    except ImportError:
        pass

    t0 = time(); print('loading bm25 index...')
    bm25_index = LuceneSearcher(kwargs['index_path'])
    t1 = time(); print('loading bm25 index...done ({:.2f} sec)'.format(t1-t0))

    return bm25_index, kwargs

def initialize_faiss(kwargs):

    t0 = time(); print('loading faiss index...')
    vector_index = faiss.read_index(kwargs['index_path'])
    t1 = time(); print('loading faiss index...done ({:.2f} sec)'.format(t1-t0))

    return vector_index, kwargs

def initialize_diskann(kwargs):

    try:
        import diskannpy
    except ImportError:
        pass

    t0 = time(); print('loading DiskANN index...')
    vector_index = diskannpy.StaticDiskIndex(
        index_directory=os.path.abspath(kwargs['index_directory']),
        index_prefix=kwargs['index_prefix'],
        num_nodes_to_cache=0,
        cache_mechanism=2,
        distance_metric='mips',
        vector_dtype=numpy.float32,
        num_threads=1,
        dimensions=1024
    )
    t1 = time(); print('loading DiskANN index...done ({:.2f} sec)'.format(t1-t0))
    return vector_index, kwargs

def initialize_vector_index(kwargs):

    if kwargs['type'] == 'bm25':
        return initialize_bm25(kwargs)
    elif kwargs['type'] == 'faiss':
        return initialize_faiss(kwargs)
    elif kwargs['type'] == 'diskann':
        return initialize_diskann(kwargs)
    else:
        raise NotImplementedError('type should be "faiss", "bm25", or "diskann".')
