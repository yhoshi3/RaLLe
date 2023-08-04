# Copyright (c) Kioxia corporation and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os, shutil
import json
import mlflow
from mlflow import log_metric, log_artifacts

from ralle.evaluation.eval_kilt_plus_has_answer import evaluate

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
            if 'output' in v:
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

def calc_metric(args):

    reduce_results(args.output_path)

    results = evaluate(args.qa_path, 'outputs/output_reduced.jsonl', -1)
    print(results)

    for k, v in results.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                log_metric(kk, vv)
        else:
            log_metric(k, v)

    os.makedirs('outputs', exist_ok=True)
    with open("outputs/results.json", "w") as f:
        json.dump(results, f, indent=4)
    
    shutil.copy(args.output_path, 'outputs/output.txt')
    shutil.copy('outputs/output.jsonl', 'outputs/output.txt')
    shutil.copy('outputs/output_reduced.jsonl', 'outputs/output_reduced.txt')

    log_artifacts("outputs")

if __name__ == "__main__":

    parser = argparse.ArgumentParser("run evaluate")
    parser.add_argument("--output_path")
    parser.add_argument("--qa_path")
    parser.add_argument("--experiment", default='Default')
    parser.add_argument("--run_name", default='tmp')
    args = parser.parse_args()

    mlflow.set_experiment(args.experiment)

    with mlflow.start_run(run_name="{}".format(args.run_name)):
        calc_metric(args)
