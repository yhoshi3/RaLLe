# Copyright (c) Kioxia corporation and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import copy
import os, shutil
import json
import subprocess
from mlflow import log_metric, log_param, log_artifacts
import numpy
import sys
import re
from time import time, sleep
from tqdm import tqdm
from ralle.evaluation.eval_kilt_plus_has_answer import evaluate
from ralle.utils import read_config, load_llms, load_retrievers

def flex_format_batch(template, history_org, eval_mode):

    used_history = {}
    cnt_used_key = 0

    for k, v in history_org.items():
        if k in template:
            cnt_used_key += 1
            if k in ['prompt', 'response', 'wiki_id_title']:
                # transpose
                v = [list(x) for x in zip(*v)]
            used_history[k] = v

    used_history_mod = [{k: v[i] for k, v in used_history.items()} for i in range(len(history_org['question']))]
    if eval_mode == 'eval':
        out = []
        for i in range(len(history_org['question'])):
            for k, v in used_history_mod[i].items():
                if k == 'question':
                    question = v
                elif k == 'prompt':
                    prompt = v
                elif k == 'response':
                    response = v
                elif k == 'wiki_id_title':
                    wiki_id_title = v
                else:
                    raise NotImplementedError
            out.append(eval(template))
    elif eval_mode == 'f-strings':
        out = [template.format(**(used_history_mod[i])) for i in range(len(history_org['question']))]
    else:
        raise ValueError
    
    return out

def reduce_results(filename):
    data = []
    with open(filename, "r") as fin:
        lines = fin.readlines()
        for line in lines:
            data.append(json.loads(line))

    with open('outputs/output_reduced.jsonl', 'w') as fout:
        for v in data:
            out_l = {}
            out_l['id'] = v['id']
            out_l['output'] = [{'provenance': []}]
            found_answer = False
            for vv in v["output"]:
                if 'provenance' in vv:
                    for vvv in vv['provenance']:
                        out_l['output'][0]['provenance'].append({"wikipedia_id": vvv['wikipedia_id'], "doc_id": vvv['doc_id']})
                elif 'answer' in vv:
                    out_l['output'][0]['answer'] = vv['answer']
                    found_answer = True
            if not found_answer:
                out_l['output'][0]['answer'] = ''
            json.dump(out_l, fout)
            fout.write('\n')
            fout.flush()

def evaluate_ralle(config_exp, config_base, funcs=None):

    if os.path.exists('outputs'):
        if 'resume' in config_exp['chain_config']['dataset']:
            f_option = 'a'
            with open('outputs/output.jsonl', f_option) as fout:
                json.dump({}, fout)
                fout.write('\n')

        else:
            f_option = 'w'
            shutil.rmtree('outputs')
    else:
        f_option  = 'w'

    os.makedirs('outputs', exist_ok=True)

    qa_path = config_base['dataset_config']['test'][config_exp['chain_config']['dataset']['dataset_name']]

    batch_size = config_exp['chain_config']['dataset']['batch_size']

    if funcs is None:
        funcs = {}
        used_llm = {}
        used_ret = {}
        for v in config_exp['chain_config']['chain']:
            if v['function'] == 'LLM':
                used_llm = v['llm_name']
            elif v['function'] == 'Retriever':
                used_ret = v['retriever_name']

        load_llms(used_llm, config_base=config_base, loaded_llm={}, funcs=funcs)
        load_retrievers(used_ret=used_ret, config_base=config_base, loaded_ret={}, funcs=funcs)

    if config_exp['chain_config']['dataset']['num_evaluate'] > 0:
        num_qa = config_exp['chain_config']['dataset']['num_evaluate']
    else:
        num_qa = int(subprocess.check_output(['wc', '-l', qa_path]).decode().split(' ')[0])
    print(num_qa)

    t_start = time()

    with open('outputs/output.jsonl', f_option) as fout:
        with open(qa_path, "r") as fin:
            cnt_batch = 0
            qa_batch = []
            out = []
            for cnt, line in zip(tqdm(range(num_qa)), fin):
                if 'resume' in config_exp['chain_config']['dataset']:
                    if cnt <= config_exp['chain_config']['dataset']['resume']:
                        continue
                qa = json.loads(line)
                qa_batch.append(qa)
                out.append({'id': qa['id'], 'input': qa['input'], 'output': []})
                if cnt_batch != int(num_qa // batch_size) and (cnt + 1) % batch_size != 0:
                    continue
                elif cnt_batch == int(num_qa // batch_size) and cnt != num_qa - 1:
                    continue
                cnt_batch += 1
                actual_batch_size = len(out)

                # Chain version
                questions = [qa['input'] for qa in qa_batch]
                history = {}
                history['question'] = questions
                history['prompt'] = []
                history['response'] = []
                history['wiki_id_title'] = []
                for c, c_cfg in enumerate(config_exp['chain_config']['chain']):
                    if config_exp['chain_config']['chain'][c]['prompt_template'] == '':
                        break
                    textin = flex_format_batch(c_cfg['prompt_template'], history, c_cfg['f-strings_or_eval'])
                    history['prompt'].append(textin)
                    for cnt_i in range(actual_batch_size):
                        out[cnt_i]['output'].append({'prompt': textin[cnt_i]})

                    if c_cfg['function'] == 'Retriever':
                        _, doc_id, wids, title, text = funcs['retrievers'][c_cfg['retriever_name']].search(textin, c_cfg['npassage'])
                        for cnt_i in range(actual_batch_size):
                            out[cnt_i]['output'][c]['provenance'] \
                                = [{'wikipedia_id': wid, 'wikipedia_title': tit, 'doc_id': did, 'text': tx} for wid, did, tit, tx in zip(wids[cnt_i], doc_id[cnt_i], title[cnt_i], text[cnt_i])]
                        answer = ['; '.join([t for t in text[iq]]) for iq in range(len(questions))]
                        wiki_ids = ['; '.join([tit + ', ' + str(ids) for tit, ids in zip(title[iq], wids[iq])]) for iq in range(len(questions))]
                    else:
                        if config_exp['chain_config']['chain'][c]['function'] == 'LLM':
                            answer = funcs['llms'][c_cfg['llm_name']](textin)
                        elif config_exp['chain_config']['chain'][c]['function'] == 'Identity':
                            answer = textin
                        else:
                            raise NotImplementedError
                        for cnt_i in range(actual_batch_size):
                            out[cnt_i]['output'][c]['answer'] = answer[cnt_i]
                        wiki_ids = ['' for _ in range(len(questions))]

                    history['response'].append(answer)
                    history['wiki_id_title'].append(wiki_ids)
                    # print('c: {}, textin: {}, answer: {}'.format(c, textin, answer))

                    if c == config_exp['chain_config']['len_chain']:
                        break

                for out_i in out:
                    json.dump(out_i, fout)
                    fout.write('\n')
                fout.flush()

                qa_batch = []
                out = []

                # if cnt==1:
                #     break

    t_end = time()


    reduce_results('outputs/output.jsonl')

    results = evaluate(qa_path, 'outputs/output_reduced.jsonl', num_qa)
    print(results)

    for k, v in results.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                log_metric(kk, vv)
        else:
            log_metric(k, v)

    log_metric("sec per query", (t_end - t_start) / num_qa)

    os.makedirs('outputs', exist_ok=True)
    with open("outputs/results.json", "w") as f:
        json.dump(results, f, indent=4)

    with open("outputs/config.json", "w") as f:
        json.dump(config_exp, f, indent=4, sort_keys=True)

    with open('outputs/prompt.txt', 'w') as f:
        for i in range(config_exp['chain_config']['len_chain']):
            f.write('== prompt_template[i] ==\n{}\n'.format(i, config_exp['chain_config']['chain'][i]['prompt_template']))
    
    shutil.copy('outputs/output.jsonl', 'outputs/output.txt')
    shutil.copy('outputs/output_reduced.jsonl', 'outputs/output_reduced.txt')

    log_artifacts("outputs")

if __name__ == "__main__":

    parser = argparse.ArgumentParser("run evaluate")
    parser.add_argument("--config", default='scripts/configs/experiment_settings/default.json', help="configuration of retrieval-augmented LLM for evaluateion")
    parser.add_argument("--config_llms", default='scripts/configs/base_settings/llms.json', help="path to llm config file")
    parser.add_argument("--config_retrievers", default='scripts/configs/base_settings/retrievers.json', help="path to retriever config file")
    parser.add_argument("--config_query_encoders", default='scripts/configs/base_settings/query_encoders.json', help="path to query encoders config file")
    parser.add_argument("--config_corpora", default='scripts/configs/base_settings/corpora.json', help="path to corpora config file")
    parser.add_argument("--config_datasets", default='scripts/configs/base_settings/datasets.json', help="path to datasets config file")
    parser.add_argument("--debug", action='store_true')
    args = parser.parse_args()

    config_exp, config_base = read_config(args)

    evaluate_ralle(config_exp, config_base)
