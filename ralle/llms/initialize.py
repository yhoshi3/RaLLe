# Copyright (c) Kioxia corporation and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import importlib
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, TextIteratorStreamer
import torch

def initialize_llm(model_args={}, pipeline_args={}):

    if model_args['model_type'] == 'openai':
        import openai
        import os

        openai.api_type = os.getenv("AZURE_OPENAI_API_TYPE")
        openai.api_base = os.getenv("AZURE_OPENAI_API_BASE") 
        openai.api_version = os.getenv("AZURE_OPENAI_API_VERSION")
        openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")

        openai.proxy = {
            "http": os.getenv("AZURE_OPENAI_PROXY"),
            "https": os.getenv("AZURE_OPENAI_PROXY")
        }
        # os.environ["http_proxy"] = os.getenv("AZURE_OPENAI_PROXY")
        # os.environ["https_proxy"] = os.getenv("AZURE_OPENAI_PROXY")

        def generate_fn(prompts):

            if isinstance(prompts, str):
                print('single mode llm function is called')
                response = openai.ChatCompletion.create(
                    engine="gpt-35-turbo",
                    max_tokens=1000,
                    messages=[
                        {"role": "user", "content": prompts}
                    ]
                )
                # return '{}: {}'.format(response['usage']['total_tokens'], response['choices'][0]['message']['content'])
                return response['choices'][0]['message']['content']

            elif isinstance(prompts, list):
                print('batch mode llm function is called')
                results = []
                for p in prompts:
                    res = openai.ChatCompletion.create(
                        engine="gpt-35-turbo",
                        max_tokens=1000,
                        messages=[
                            {"role": "user", "content": p}
                        ]
                    )
                    print('openai response: {}'.format(res))
                    # out = '{}: {}'.format(res['usage']['total_tokens'], res['choices'][0]['message']['content'])
                    out = res['choices'][0]['message']['content']
                    results.append(out) 
                return results
            else:
                raise ValueError
        
        return generate_fn, None

    else:
        quantization_module = importlib.import_module('.'.join(model_args['quantization_config']['quantization_type'].split('.')[:-1]))
        quantization_class = getattr(quantization_module, model_args['quantization_config']['quantization_type'].split('.')[-1])
        quantization_config_config = model_args['quantization_config']['quantization_config_config']
        quantization_config = quantization_class(**quantization_config_config)

        print('Loading Tokenizer...')
        tokenizer = AutoTokenizer.from_pretrained(model_args['model_type'])
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        model_kwargs = model_args.copy()
        del model_kwargs['model_type'], model_kwargs['device_map'], model_kwargs['quantization_config']
        print('Loading LLM model...')
        model = AutoModelForCausalLM.from_pretrained(
                    model_args['model_type'],
                    device_map=model_args['device_map'],
                    quantization_config=quantization_config,
                    torch_dtype=torch.float16,
                    **model_kwargs)

        model.eval()

        pipe = pipeline("text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    streamer=streamer,
                    **pipeline_args)

        def generate_fn(prompts, **kwargs):

            if isinstance(prompts, str):
                print('single mode llm function is called')
                return pipe([prompts], **kwargs)[0][0]["generated_text"][len(prompts) :]
            elif isinstance(prompts, list):
                print('batch mode llm function is called')
                outputs = pipe(prompts, **kwargs)
                results = [out[0]["generated_text"][len(prompt) :] for prompt, out in zip(prompts, outputs)]
                return results
            else:
                raise ValueError

        return generate_fn, streamer
