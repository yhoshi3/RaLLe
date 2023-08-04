# Copyright (c) Kioxia corporation and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import importlib
from functools import partial
from transformers import AutoModel, AutoTokenizer
from .encode_functions.e5 import encode

def initialize_query_encoder(kwargs):

    if 'quantization_config' in kwargs and kwargs['quantization_config'] is not None:
        use_quantization = True
        quantization_module = importlib.import_module('.'.join(kwargs['quantization_config']['quantization_type'].split('.')[:-1]))
        quantization_class = getattr(quantization_module, kwargs['quantization_config']['quantization_type'].split('.')[-1])
        quantization_config_config = kwargs['quantization_config']['quantization_config_config']
        quantization_config = quantization_class(**quantization_config_config)
    else:
        use_quantization = False

    tokenizer = AutoTokenizer.from_pretrained(kwargs['model_type'])
    if use_quantization:
        model = AutoModel.from_pretrained(kwargs['model_type'], device_map=kwargs['device_map'], quantization_config=quantization_config)
    else:
        model = AutoModel.from_pretrained(kwargs['model_type'])
        model.to('cuda:0')
    model.eval()

    encode_args = {'tokenizer': tokenizer, 'model': model}
    encode_fn = partial(encode, **encode_args)

    return encode_fn
