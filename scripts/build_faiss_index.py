# Copyright (c) Kioxia corporation and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy
import faiss
import os
import argparse
from time import time

parser = argparse.ArgumentParser()
parser.add_argument('--embeddings', type=str, default='', help='path to embedding data')
parser.add_argument('--subset', type=int, default=0, help='use only the first [value] vectors')
parser.add_argument('--index_type', type=str, choices=['flat', 'hnsw'], default='flat', help='index type')
parser.add_argument('--m', type=int, default=16, help='faiss hnsw M parameter')
parser.add_argument('--ef_construction', type=int, default=40, help='faiss hnsw efConstruction parameter')
parser.add_argument('--faiss_threads', type=int, default=16, help='faiss omp nummber of threads')

args = parser.parse_args()
print(args)

faiss.omp_set_num_threads(args.faiss_threads)

name_list = os.path.splitext(os.path.basename(args.embeddings))[0].split('_')
ext = os.path.splitext(os.path.basename(args.embeddings))[1]

dim = int(name_list[-2])
num_keys = int(name_list[-1])
print('dim: {}, num_keys: {}'.format(dim, num_keys))

dir_prefix = os.path.dirname(args.embeddings)
output_dir = os.path.join(dir_prefix, 'index')
if args.index_type == 'flat':
    output_dir = os.path.join(output_dir, 'faiss-flat')
elif args.index_type == 'hnsw':
    output_dir = os.path.join(output_dir, 'faiss-hnsw')
else:
    raise NotImplementedError
os.makedirs(output_dir, exist_ok=True)

index_path = ''
if args.subset > 0:
    index_path += '{}_'.format('_'.join([nnn.replace('full', 'subset') for nnn in name_list[:-2]]))
else:
    index_path += '{}_'.format('_'.join(name_list[:-2]))
if args.index_type == 'hnsw':
    index_path += 'm{}_efc{}_'.format(args.m, args.ef_construction)
if args.subset > 0:
    index_path += '{}_{}.index'.format(dim, args.subset)
else:
    index_path += '{}_{}.index'.format(dim, num_keys)
index_path = os.path.join(output_dir, index_path)
print(index_path)

t0 = time()
print('loading embeddings...')
if ext == '.npmmap':
    if args.subset > 0:
        key = numpy.memmap(args.embeddings, dtype=numpy.float32, mode='r', shape=(num_keys, dim))[:args.subset].astype(numpy.float32)
    else:
        key = numpy.memmap(args.embeddings, dtype=numpy.float32, mode='r', shape=(num_keys, dim)).astype(numpy.float32)
else:
    if args.subset > 0:
        key = numpy.load(args.embeddings)[:args.subset].astype(numpy.float32)
    else:
        key = numpy.load(args.embeddings).astype(numpy.float32)

t1 = time()
print('loading embeddings...done({:.2f}sec).'.format(t1-t0))

if args.index_type == 'flat':
    index = faiss.IndexFlat(key.shape[1], faiss.METRIC_INNER_PRODUCT)
elif args.index_type == 'hnsw':
    index = faiss.IndexHNSWFlat(dim, args.m, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = args.ef_construction
else:
    raise NotImplementedError

t0 = time()
print('indexing...')
index.add(key)
t1 = time()
print('indexing...done({:.2f}sec).'.format(t1-t0))
print('write {}...'.format(index_path))
t0 = time()
faiss.write_index(index, index_path)
t1 = time()
print('write {}...done({:.2f}sec).'.format(index_path, t1-t0))

