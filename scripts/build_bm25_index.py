# Copyright (c) Kioxia corporation and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import os
import subprocess

def read_file(infile, handle_file, log=False, skip_first_line=False):
    if log:
        print('Opening "{}"...'.format(infile))
    data = None
    with open(infile) as f:
        if skip_first_line:
            f.readline()
        data = handle_file(f)
    if log:
        print('  Done.')
    return data

def read_tsv(infile, row_fn=lambda x: x, log=False, skip_first_line=False):
    handler = lambda f: [row_fn(line.split('\t')) for line in f.readlines()]
    return read_file(infile, handler, log=log, skip_first_line=skip_first_line)

def write_file(outfile, handle_file, log=False):
    if log:
        print('Writing to "{}"...'.format(outfile))
    with open(outfile, 'w+') as f:
        handle_file(f)
    if log:
        print('  Done.')

def write_json(outfile, data, log=False, pretty=False):
    handler = lambda f: f.write(json.dumps(data, indent=4 if pretty else None))
    write_file(outfile, handler, log=log)

def prepare_passages_to_index(wikipath):
    index_path = '/'.join(['bm25' if v == 'text' else v for v in wikipath.split('/')])
    index_path = os.path.splitext(index_path)[0]
    outdir = os.path.join(os.path.dirname(wikipath), 'bm25_preprocessed', os.path.splitext(os.path.basename(wikipath))[0])
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, 'preprocessed_corpus.json')
    print(outpath)
    def _serialize(split):
        pid = split[0].strip().lower()
        title = split[2].strip()
        text = split[1].strip()
        return { 'id': pid, 'contents': f'{title}, {text}' }
    passages = read_tsv(wikipath, row_fn=_serialize, log=True)
    write_json(outpath, passages, log=True)
    print('Total number of passages: {}'.format(len(passages)))
    print('Run the following command to build bm25 index')
    command = f'time python -m pyserini.index -collection JsonCollection -generator DefaultLuceneDocumentGenerator -threads 4 -input {outdir} -index {index_path} -storePositions -storeDocvectors -storeRaw'
    return command

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--wiki_passages_file', required=True, type=str, default=None,
                        help="Location of the Wikipedia passage splits.")
    args = parser.parse_args()

    command = prepare_passages_to_index(args.wiki_passages_file)
    subprocess.run([v for v in command.split(' ')])
