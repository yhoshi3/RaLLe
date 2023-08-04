# Copyright (c) Kioxia corporation and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
from ralle.llms.initialize import initialize_llm
from ralle.retrievers import initialize_corpus, initialize_query_encoder, initialize_retriever

def read_config(args):

    with open(args.config, "r", encoding="utf-8") as reader:
        text = reader.read()
        config_exp = json.loads(text)

    config_base = {}
    with open(args.config_system, "r", encoding="utf-8") as reader:
        text = reader.read()
        config_base['system_config'] = json.loads(text)

    with open(args.config_llms, "r", encoding="utf-8") as reader:
        text = reader.read()
        config_base['llm_config'] = json.loads(text)

    with open(args.config_retrievers, "r", encoding="utf-8") as reader:
        text = reader.read()
        config_base['retriever_config'] = json.loads(text)

    with open(args.config_query_encoders, "r", encoding="utf-8") as reader:
        text = reader.read()
        config_base['query_encoder_config'] = json.loads(text)

    with open(args.config_corpora, "r", encoding="utf-8") as reader:
        text = reader.read()
        config_base['corpus_config'] = json.loads(text)

    with open(args.config_datasets, "r", encoding="utf-8") as reader:
        text = reader.read()
        config_base['dataset_config'] = json.loads(text)

    return config_exp, config_base


def load_llms(used_llm, config_base=None, loaded_llm=None, funcs=None):

    to_be_loaded = (set(used_llm) | loaded_llm) - loaded_llm
    if len(to_be_loaded) != 0:
        funcs['llms'] = {}
        funcs['streamer'] = {}
        for llm in to_be_loaded:
            funcs['llms'][llm], funcs['streamer'][llm] = initialize_llm(**config_base['llm_config'][llm])
    
    loaded_llm |= to_be_loaded
    print('model load finished!')

def load_retrievers(used_ret, config_base=None, loaded_ret=None, funcs=None):

    to_be_loaded = (set(used_ret) | loaded_ret) - loaded_ret
    print('debug: to_be_loaded: {}'.format(to_be_loaded))

    if len(to_be_loaded) != 0:
        used_qe = set()
        used_cor = set()
        for ret_cfg in config_base['retriever_config'].keys():
            if ret_cfg in to_be_loaded:
                if config_base['retriever_config'][ret_cfg]['type'] in ('faiss', 'diskann'):
                    used_qe.add(config_base['retriever_config'][ret_cfg]['query_encoder_name'])
                used_cor.add(config_base['retriever_config'][ret_cfg]['corpus_name'])

        query_encoders = {}
        if len(used_qe) != 0:
            for emb in config_base['query_encoder_config'].keys():
                if emb in used_qe:
                    print('initialize query encoder: {}'.format(config_base['query_encoder_config'][emb]['model_type']))
                    query_encoders[emb] = initialize_query_encoder(config_base['query_encoder_config'][emb])

        corpora = {}
        if len(used_cor) != 0:
            for cor in config_base['corpus_config'].keys():
                if cor in used_cor:
                    print('initialize corpus: {}'.format(config_base['corpus_config'][cor]['corpus_path']))
                    corpora[cor] = initialize_corpus(config_base['corpus_config'][cor])

        funcs['retrievers'] = {}
        for ret_cfg in config_base['retriever_config'].keys():
            if ret_cfg in to_be_loaded:
                print('initialize index')
                funcs['retrievers'][ret_cfg] = initialize_retriever(config_base['retriever_config'][ret_cfg], query_encoders, corpora)

    loaded_ret |= to_be_loaded
    print('model load finished!')


def flex_format(text, history):
    used_history = {}
    for k, v in history.items():
        if k in text:
            used_history[k] = v
    
    return text.format(**used_history)


def interpret_prompt(format_prompt, prompt_template, step=0, history=None):

    if format_prompt == 'f-strings':
        # example: {question}, {response[0]}
        t_in = flex_format(prompt_template, history)
    elif format_prompt == 'eval':
        # example: "{}".format(question), "{}".format(response[0])
        question = history['question']
        prompt = history['prompt']
        response = history['response']
        wiki_id_title = history['wiki_id_title']
        t_in = eval(prompt_template)

    history['prompt'][step] = t_in

    return t_in


def exec_function(function, llm, retriever, k, format_prompt, prompt_template, step=None, history=None, funcs=None):
    if prompt_template == '':
        # do nothing
        return '', '', ''

    else:
        t_in = interpret_prompt(format_prompt, prompt_template, step=step, history=history)

        if function == 'LLM':
            t_out = funcs['llms'][llm](t_in)
            wiki_ids = ''
        elif function == 'Retriever':
            _, _, wids, title, t_out = funcs['retrievers'][retriever].search(t_in, int(k))
            wiki_ids = '; '.join([tit + ', ' + str(ids) for tit, ids in zip(title[0], wids[0])])
            t_out = '; '.join([t for t in t_out[0]])
        elif function == 'Identity':
            t_out = t_in
            wiki_ids = ''
        
        history['response'][step] = t_out
        history['wiki_id_title'][step] = wiki_ids

        return t_in, t_out, wiki_ids
