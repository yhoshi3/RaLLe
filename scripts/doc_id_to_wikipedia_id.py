# Copyright (c) Kioxia corporation and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import pickle
from tqdm import tqdm

parser = argparse.ArgumentParser(description='make doc_id to wikipedia_id mapping')
parser.add_argument('--corpus_path', default='data/text/corpus/kilt/kilt_w100_title.tsv',
                    help='path to corpus file')
parser.add_argument('--mapping_path', default='data/text/corpus/kilt/mapping_KILT_title.p',
                    help='path to mapping file')
parser.add_argument('--number_of_questions', type=int, default=100,
                    help='the number of questions to be picked up')
parser.add_argument('--output_path', default='data/text/corpus/kilt/doc_id_to_wikipedia_id_mapping.p',
                    help='path to output dataset file')

args = parser.parse_args()


print('Loading the corpus...')
with open(args.corpus_path) as file_in:
    corpus = [v.strip().split('\t') for v in file_in]

corpus_doc_id = [c[0] for c in corpus]
corpus_title = [c[2] for c in corpus]

with open(args.mapping_path, 'rb') as file_in:
    title_to_wikipedia_id_mapping = pickle.load(file_in)

output = {}
for doc_id, title in zip(tqdm(corpus_doc_id), corpus_title):
    if not doc_id.isdecimal():
        continue

    if title in title_to_wikipedia_id_mapping:
        output[int(doc_id)] = int(title_to_wikipedia_id_mapping[title])
    elif title.replace('"', '') in title_to_wikipedia_id_mapping:
        output[int(doc_id)] = int(title_to_wikipedia_id_mapping[title.replace('"', '')])
    else:
        output[int(doc_id)] = -1 # can not find title in mapping file

with open(args.output_path, 'wb') as fout:
    pickle.dump(output, fout)
