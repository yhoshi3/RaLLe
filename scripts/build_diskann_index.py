# Copyright (c) Kioxia corporation and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import diskannpy
import os
import argparse
from time import time

default_params = {
    'l': 125,
    'pq_vector_size': 512,
    'degree': 64,
    'build_memory_budget': 64
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--npmmap_file', type=str, help='Path to npmmap file')
    parser.add_argument('--index_dir', type=str, help='directory where DiskANN index files are created')
    parser.add_argument('--index_prefix', type=str, default='index', help='Prefix for DiskANN index file names (default: index)')
    parser.add_argument('--num_threads', '-T', type=int, default=1, help='Number of thread used for index construction')
    args = parser.parse_args()

    data_type = np.float32
    component_size = np.dtype(data_type).itemsize
    dataset_file = args.npmmap_file
    dataset_file_size = os.path.getsize(dataset_file)
    ndim = int(os.path.basename(dataset_file).split('_')[-2])
    npts, rem = divmod(dataset_file_size, ndim * component_size)
    assert rem == 0, 'Size of embeddings file is incorrect, may be corrupted.'
    dataset = np.memmap(dataset_file, dtype=data_type, mode='readonly', shape=(npts, ndim))
    print(f'Dataset file has {npts} of {ndim}-dimensional vectors.')
    search_memory_budget = (npts * default_params['pq_vector_size'] + 250000 * (4 * default_params['degree'] + component_size * ndim)) / (2 ** 30)

    os.makedirs(args.index_dir, exist_ok=True)
    time_build_start = time()
    diskannpy.build_disk_index(
        num_threads=args.num_threads,
        data=dataset,
        vector_dtype=data_type,
        distance_metric='mips',
        index_directory=args.index_dir,
        index_prefix=args.index_prefix,
        complexity=default_params['l'],
        graph_degree=default_params['degree'],
        search_memory_maximum=search_memory_budget,
        build_memory_maximum=default_params['build_memory_budget']
    )

    time_build_end = time()
    print('Index build and write finished ({:.2f} s).'.format(time_build_end - time_build_start))

if __name__ == '__main__':
    main()
