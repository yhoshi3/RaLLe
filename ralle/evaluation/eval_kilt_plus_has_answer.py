# Copyright (c) Kioxia corporation and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import pprint

from kilt import eval_retrieval as retrieval_metrics
from kilt import kilt_utils
from kilt.eval_downstream import normalize_answer, get_gold_answers, _metric_max_over_ground_truths, validate_input, _calculate_metrics

# answer in output
def _has_answer(prediction, ground_truth):
    return normalize_answer(ground_truth) in normalize_answer(prediction)


def calculate_has_answer_metric(gold_records, guess_records, outputs):

    total_count = 0

    # downstream metrics
    has_answer = 0

    # kilt metrics
    kilt_has_answer = 0

    for guess_item, gold_item in zip(guess_records, gold_records):

        gold_candidate_answers = get_gold_answers(gold_item)
        guess_answer = str(guess_item["output"][0]["answer"]).strip()

        if len(guess_answer) == 0:
            # empty answer
            continue

        total_count += 1
        local_aio = _metric_max_over_ground_truths(
            _has_answer, guess_answer, gold_candidate_answers
        )
        has_answer += local_aio

        # KILT-metrics
        Rprec = retrieval_metrics.rprecision(
            guess_item, gold_item, rank_keys=["wikipedia_id"]
        )
        if Rprec == 1:
            kilt_has_answer += local_aio

    if total_count > 0:
        has_answer /= total_count
        kilt_has_answer /= total_count

    outputs['kilt']['KILT-has_answer'] = kilt_has_answer
    outputs['downstream']['has_answer'] = has_answer

    return outputs

def evaluate(gold, guess, num=-1):
    pp = pprint.PrettyPrinter(indent=4)

    gold_records = kilt_utils.load_data(gold)
    guess_records = kilt_utils.load_data(guess)
    if num>0:
        gold_records = gold_records[:num]
        guess_records = guess_records[:num]

    # 0. validate input
    gold_records, guess_records = validate_input(gold_records, guess_records)

    # 1. downstream + kilt
    result = _calculate_metrics(gold_records, guess_records)
    result = calculate_has_answer_metric(gold_records, guess_records, result)

    # 2. retrieval performance
    retrieval_results = retrieval_metrics.compute(
        gold_records, guess_records, ks=[1, 5], rank_keys=["wikipedia_id"]
    )
    result["retrieval"] = {
        "Rprec": retrieval_results["Rprec"],
        "recall_at_5": retrieval_results["recall@5"],
    }

    pp.pprint(result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("guess", help="Guess KILT file")
    parser.add_argument("gold", help="Gold KILT file")

    args = parser.parse_args()
    evaluate(args.gold, args.guess)
