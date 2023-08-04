# Copyright (c) Kioxia corporation and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import os
import pickle

parser = argparse.ArgumentParser(description='question pickup')
parser.add_argument('--dataset_path', default='KILT/data/nq-train-kilt.jsonl',
                    help='path to input dataset file')
parser.add_argument('--corpus_path', default='data/text/corpus/kilt/kilt_w100_title_modified_499992.tsv',
                    help='path to corpus file')
parser.add_argument('--mapping_path', default='data/text/corpus/kilt/mapping_KILT_title.p',
                    help='path to mapping file')
parser.add_argument('--number_of_questions', type=int, default=100,
                    help='the number of questions to be picked up')
parser.add_argument('--output_path', default='data/text/query/kilt/nq-train-kilt_100.jsonl',
                    help='path to output dataset file')

args = parser.parse_args()

with open(args.dataset_path, "r") as fin:
    data = [json.loads(line) for line in fin]

with open(args.corpus_path) as file_in:
    corpus_title = [v.strip().split('\t')[2] for v in file_in]

with open(args.mapping_path, 'rb') as file_in:
    kilt_mapping = pickle.load(file_in)

used_wikipedia_id = set()
not_found_title = set()
for title in corpus_title:
    if title in kilt_mapping:
        wikipedia_id = int(kilt_mapping[title])
        used_wikipedia_id.add(wikipedia_id)
    elif title.replace('"', '') in kilt_mapping:
        wikipedia_id = int(kilt_mapping[title.replace('"', '')])
        used_wikipedia_id.add(wikipedia_id)
    else:
        if title not in not_found_title:
            print('{} was not found in the mapping file'.format(title))
            not_found_title.add(title)

print(len(used_wikipedia_id))

cnt = 0

os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
with open(args.output_path, 'w', encoding='utf-8') as of:
    for v in data:
        if cnt == args.number_of_questions:
            break
        for vv in v['output']:
            if 'provenance' in vv:
                flg1 = False
                for vvv in vv['provenance']:
                    if int(vvv['wikipedia_id']) in used_wikipedia_id:
                        json.dump(v, of, ensure_ascii=False)
                        of.write('\n')
                        cnt += 1
                        flg1 = True
                        break
                if flg1:
                    break
