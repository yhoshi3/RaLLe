# Copyright (c) Kioxia corporation and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import glob
import json
import os
import mlflow
from ralle.evaluation.evaluate import evaluate_ralle
from ralle.utils import read_config, load_llms, load_retrievers

def experiment(args):

    _, config_base = read_config(args)
    mlflow.set_experiment(args.experiment)

    cwd = os.getcwd()
    os.chdir(args.config_exp_path)
    config_paths = glob.glob("./**/*.json", recursive=True)
    config_paths = sorted(config_paths)
    config_paths = [os.path.join(args.config_exp_path, os.path.basename(c)) for c in config_paths]
    os.chdir(cwd)

    print(config_paths)

    funcs = {}
    used_llm = set()
    used_ret = set()

    tmp_cfg_path = config_paths[0]
    with open(tmp_cfg_path, "r", encoding="utf-8") as reader:
        text = reader.read()
    config_exp = json.loads(text)
    for v in config_exp['chain_config']['chain']:
        if 'llm_name' in v:
            used_llm.add(v['llm_name'])
        if 'retriever_name' in v:
            used_ret.add(v['retriever_name'])
    
    used_llm = list(used_llm)
    used_ret = list(used_ret)

    assert len(used_llm) <= 1, "multi llms are not supported in this script"
    assert len(used_ret) <= 1, "multi retrievers are not supported in this script"

    if len(used_llm) == 1 and args.llm_overwrite != '':
        used_llm[0] = args.llm_overwrite
    if len(used_ret) == 1 and args.retriever_overwrite != '':
        used_ret[0] = args.retriever_overwrite

    load_llms(used_llm, config_base=config_base, loaded_llm=set(), funcs=funcs)
    load_retrievers(used_ret, config_base=config_base, loaded_ret=set(), funcs=funcs)

    for cfg_path in config_paths:
        with open(cfg_path, "r", encoding="utf-8") as reader:
            text = reader.read()
        config_exp = json.loads(text)

        # For overwriting llm
        for v in config_exp['chain_config']['chain']:
            if args.llm_overwrite != '' and 'llm_name' in v:
                v['llm_name'] = used_llm[0]
            if args.retriever_overwrite != '' and 'retriever_name' in v:
                v['retriever_name'] = used_ret[0]
            if args.npassage_overwrite > 0 and 'npassage' in v:
                v['npassage'] = args.npassage_overwrite

        if args.batch_size > 0:
            config_exp['chain_config']['dataset']['batch_size'] = args.batch_size

        with mlflow.start_run(run_name="{}".format(os.path.splitext(os.path.basename(cfg_path))[0])):
            evaluate_ralle(config_exp, config_base, funcs=funcs)


if __name__ == "__main__":

    parser = argparse.ArgumentParser("run evaluate")
    parser.add_argument("--config", default='scripts/configs/experiment_settings/default.json', help="configuration of retrieval-augmented LLM for evaluateion")
    parser.add_argument("--config_system", default='scripts/configs/base_settings/system.json', help="path to system config file")
    parser.add_argument("--config_llms", default='scripts/configs/base_settings/llms.json', help="path to llm config file")
    parser.add_argument("--config_retrievers", default='scripts/configs/base_settings/retrievers.json', help="path to retriever config file")
    parser.add_argument("--config_query_encoders", default='scripts/configs/base_settings/query_encoders.json', help="path to query encoders config file")
    parser.add_argument("--config_corpora", default='scripts/configs/base_settings/corpora.json', help="path to corpora config file")
    parser.add_argument("--config_datasets", default='scripts/configs/base_settings/datasets.json', help="path to datasets config file")
    parser.add_argument("--experiment", default='retrieve-read_llama-2-70b')
    parser.add_argument("--config_exp_path", default='scripts/configs/experiment_settings/retrieve-read')
    parser.add_argument("--batch_size", type=int, default=-1, help='overwrite batch size')
    parser.add_argument("--llm_overwrite", default='', help='overwrite llm')
    parser.add_argument("--retriever_overwrite", default='', help='overwrite retriever')
    parser.add_argument("--npassage_overwrite", type=int, default='-1', help='overwrite retriever')

    args = parser.parse_args()

    experiment(args)
