# Copyright (c) Kioxia corporation and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import shutil
import torch
from torch import Tensor, no_grad
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
import numpy
import faiss
from tqdm import tqdm

parser = argparse.ArgumentParser(description='make embeddings')
parser.add_argument('--model', choices=['e5', 'e5-multi'], default='e5',
                    help='encoder model name')
parser.add_argument('--data_dir', default='',
                    help='directory path where kilt_w100_title.tsv exists')
parser.add_argument('--subset', type=int, default=0,
                    help='if subset<=0, process all the lines, else process only subset lines')
parser.add_argument('--col', type=int, default=1, help='column number of text')
parser.add_argument('--col_title', type=int, default=2, help='column number of title')
parser.add_argument('--use_8bit', action='store_true',
                    help='apply 8-bit quantization to encoder model parameters')
parser.add_argument('--tmp_dir', default='tmp_embs', help='output directory name')
parser.add_argument('--out_dir', default='results', help='output directory name')
parser.add_argument('--batch_size', type=int, default=100,
                    help='batch size for encoding')
parser.add_argument('--start', type=int, default=0,
                    help='start from (start * repeat * interval)th sample')
parser.add_argument('--repeat', type=int, default=1000,
                    help='number of output files to be created')
parser.add_argument('--interval', type=int, default=2000,
                    help='number of passages in an output file')
parser.add_argument('--resume', type=int, default=0,
                    help='resume from (start+resume) x interval ')

args = parser.parse_args()

# preprocess
# sprit header line and unnecessary quatations
file_processed = os.path.join(args.data_dir, 'kilt_w100_title_modified{}.tsv'.format('_subset_' + str(args.subset) if args.subset > 0 else ''))
print('file to be processed: {}'.format(file_processed))
if not os.path.exists(file_processed):
    with open(file_processed, 'w') as file_out:
        with open(os.path.join(args.data_dir, 'kilt_w100_title.tsv')) as file_in:
            for i, v in enumerate(file_in):
                if i == 0:
                    continue
                v_list = v.split('\t')
                file_out.write('{}\t{}\t{}\n'.format(v_list[0].strip(), v_list[1].strip().replace('""', '"')[1:-1], v_list[2].strip()))
                if i == args.subset:
                    break

# load preprocessed corpus file
with open(file_processed) as file_in:
    wiki_full = [v.split('\t') for v in file_in]

num_data = len(wiki_full)
print('number of passages: {}'.format(num_data))

# encoder setting
if args.model == 'e5' or args.model == 'e5-multi':

    if args.model == 'e5':
        args.model_type = 'intfloat/e5-large-v2'
    elif args.model == 'e5-multi':
        args.model_type = 'intfloat/multilingual-e5-large'

    args.dim = 1024

    def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def encode(docs, tokenizer=None, model=None):
        docs = ['passage: {}'.format(d) for d in docs]
        batch_dict = tokenizer(docs, max_length=512, padding=True, truncation=True, return_tensors='pt')
        with no_grad():
            batch_dict = batch_dict.to('cuda:0')
            outputs = model(**batch_dict)
            doc_embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

        return doc_embeddings

else:
    raise NotImplementedError("currently not supported.")

print('args: {}'.format(args))

tokenizer = AutoTokenizer.from_pretrained(args.model_type)
if args.use_8bit:
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        has_fp16_weights=False,
    )
    model = AutoModel.from_pretrained(args.model_type, device_map={'':0}, quantization_config=quantization_config)
else:
    model = AutoModel.from_pretrained(args.model_type)
    model.to('cuda:0')
model.eval()

start = args.start
os.makedirs(args.tmp_dir, exist_ok=True)
os.makedirs(args.out_dir, exist_ok=True)

batch_size = args.batch_size
write_interval = args.interval // args.batch_size
write_repeat = args.repeat
start_resume = args.resume
num_batches = len(wiki_full) // batch_size + ((len(wiki_full) % batch_size) != 0)
num_files = num_batches // write_interval + (num_batches % write_interval != 0)

print('number of batches: {}'.format(num_batches))

if (start + 1) * write_repeat * write_interval > num_batches:
    end = num_batches
    if  num_batches % write_interval != 0:
        last_one = True
    else:
        last_one = False
else:
    end = (start + 1) * write_repeat * write_interval
    last_one = False
print('processing from {} to {}'.format((start * write_repeat + start_resume) * write_interval, end))

cur = 0
for i in tqdm(range((start * write_repeat + start_resume) * write_interval, end)):

    # title and text joined by a comma
    docs = ['{}, {}'.format(v[args.col_title].strip(), v[args.col]) for v in wiki_full[i*batch_size:(i+1)*batch_size]]
    doc_embeddings = encode(docs, tokenizer=tokenizer, model=model)
    
    if i % write_interval == 0:
        if end % write_interval != 0 and i // write_interval == end // write_interval:
            if num_data % batch_size != 0:
                out = torch.zeros((batch_size * (end - write_interval * (end // write_interval) - 1) + (num_data % batch_size), args.dim), device='cuda:0')
            else:
                out = torch.zeros((batch_size * (end - write_interval * (end // write_interval)), args.dim), device='cuda:0')
        elif end % write_interval == 0 and i // write_interval == (end - 1) // write_interval:
            if num_data % batch_size != 0:
                out = torch.zeros((batch_size * (end - write_interval * ((end - 1) // write_interval) - 1) + (num_data % batch_size), args.dim), device='cuda:0')
            else:
                out = torch.zeros((batch_size * (end - write_interval * ((end - 1) // write_interval)), args.dim), device='cuda:0')
        else:
            out = torch.zeros((batch_size * write_interval, args.dim), device='cuda:0')
        out[cur:cur+len(doc_embeddings)] = doc_embeddings[:]
        cur += len(doc_embeddings)
    elif i % write_interval == write_interval - 1:
        out[cur:cur+len(doc_embeddings)] = doc_embeddings[:]
        cur += len(doc_embeddings)
        out = out.cpu().numpy()
        faiss.normalize_L2(out)
        numpy.save(os.path.join(args.tmp_dir, 'out_{:04}.npy'.format(i // write_interval)), out)
        del out
        cur = 0
    else:
        out[cur:cur+len(doc_embeddings)] = doc_embeddings[:]
        cur += len(doc_embeddings)

if last_one:
    out = out.cpu().numpy()
    faiss.normalize_L2(out)
    numpy.save(os.path.join(args.tmp_dir, 'out_{:04}.npy'.format(i // write_interval)), out)

# merge embeddings
out_shape = (num_data, args.dim)

out_file = os.path.join(args.out_dir, 'subset_{}_{}.npmmap'.format(args.dim, num_data) if args.subset > 0 
                                      else 'full_{}_{}.npmmap'.format(args.dim, num_data))

fp = numpy.memmap(out_file, dtype=numpy.float32, mode='w+', shape=out_shape)
del fp

cur = 0
for i in tqdm(range(num_files)):
    filename = os.path.join(args.tmp_dir, 'out_{:04}.npy'.format(i))
    if os.path.exists(filename):
        tmp = numpy.load(filename).astype(numpy.float32)
    else:
        print('filename is not exist: {}'.format(filename))
        tmp = numpy.zeros((args.interval, args.dim), dtype=numpy.float32)
    fp = numpy.memmap(out_file, dtype=numpy.float32, mode='r+', shape=out_shape)
    fp[cur:cur+len(tmp)] = tmp
    del fp
    cur += len(tmp)

print('Making embeddings completed!')

shutil.rmtree(args.tmp_dir)
