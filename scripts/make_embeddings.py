# Copyright (c) Kioxia corporation and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
from torch import Tensor, no_grad
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
import numpy
import faiss
from tqdm import tqdm
import linecache
import subprocess

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
parser.add_argument('--out_dir', default='results', help='output directory name')
parser.add_argument('--batch_size', type=int, default=100,
                    help='batch size for encoding')

args = parser.parse_args()

# preprocess
# sprit header line and unnecessary quotation marks
file_processed = os.path.join(args.data_dir, 'kilt_w100_title_modified{}.tsv'.format('_subset_' + str(args.subset) if args.subset > 0 else ''))
print('file to be processed: {}'.format(file_processed))
if not os.path.exists(file_processed):
    print('Loading the corpus...')
    with open(file_processed, 'w') as file_out:
        with open(os.path.join(args.data_dir, 'kilt_w100_title.tsv')) as file_in:
            for i, v in enumerate(file_in):
                if i == 0:
                    continue
                v_list = v.split('\t')
                file_out.write('{}\t{}\t{}\n'.format(v_list[0].strip(), v_list[1].strip().replace('""', '"')[1:-1], v_list[2].strip()))
                if i == args.subset:
                    break

# Dataset
def corpus_fn(ids):
    text = linecache.getline(file_processed, ids + 1)
    return text

class CorpusDataset(Dataset):
    def __init__(self, filename):
        self.corpus_fn = lambda idx: linecache.getline(filename, idx + 1)
        self.num_lines = int(subprocess.check_output(['wc', '-l', filename]).decode().split(' ')[0])

    def __len__(self):
        return self.num_lines
    
    def __getitem__(self, idx):
        return self.corpus_fn(idx)

dataset = CorpusDataset(file_processed)
num_data = len(dataset)
print('Total number of passages: {}'.format(num_data))
dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True, shuffle=False)

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

os.makedirs(args.out_dir, exist_ok=True)

out_shape = (num_data, args.dim)
out_file = os.path.join(args.out_dir, 'subset_{}_{}.npmmap'.format(args.dim, num_data) if args.subset > 0 
                                      else 'full_{}_{}.npmmap'.format(args.dim, num_data))

fp = numpy.memmap(out_file, dtype=numpy.float32, mode='w+', shape=out_shape)
del fp

for i, batch in enumerate(tqdm(dataloader, ncols=80, desc='Encoding')):
    # title and text joined by a comma
    docs = ['{}, {}'.format(v.split('\t')[args.col_title].strip(), v.split('\t')[args.col]) for v in batch]
    doc_embeddings = encode(docs, tokenizer=tokenizer, model=model)
    doc_embeddings = doc_embeddings.cpu().numpy().astype(numpy.float32)
    faiss.normalize_L2(doc_embeddings)
    fp = numpy.memmap(out_file, dtype=numpy.float32, mode='r+', shape=out_shape)
    fp[i*args.batch_size:(i+1)*args.batch_size] = doc_embeddings
    del fp
print('Creating the embeddings completed!')
