# Copyright (c) Kioxia corporation and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import faiss
import numpy
from torch import no_grad, Tensor

def average_pool(last_hidden_states: Tensor,
            attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

def encode(query, tokenizer=None, model=None):
    if isinstance(query, str):
        docs = ['query: {}'.format(query)]
    elif isinstance(query, list):
        docs = ['query: {}'.format(q) for q in query]

    batch_dict = tokenizer(docs, max_length=512, padding=True, truncation=True, return_tensors='pt')
    with no_grad():
        batch_dict = batch_dict.to('cuda:0')
        outputs = model(**batch_dict)
        doc_embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        doc_embeddings = doc_embeddings.cpu().numpy().astype(numpy.float32)
        faiss.normalize_L2(doc_embeddings)

    return doc_embeddings

