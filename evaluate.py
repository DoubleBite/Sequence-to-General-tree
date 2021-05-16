import sys
import os
import json
import argparse


def read_prediction_results(result_dir):

    predictions_path = os.path.join(result_dir, "predictions.jsonl")

    # Read the prediction results
    results = []
    with open(predictions_path, 'r') as f:
        for line in f.readlines():
            results.append(json.loads(line))

    # Calculate the number of correct equations and answers
    total_problems = len(results)
    problems_with_correct_equation = 0
    problems_with_correct_answer = 0
    for res in results:
        if res["equation_correct"] is True:
            problems_with_correct_equation += 1
        if res["answer_correct"] is True:
            problems_with_correct_answer += 1

    return {
        "correct_equation": problems_with_correct_equation,
        "correct_answer": problems_with_correct_answer,
        "total": total_problems
    }


def evaluate_performance(result_dir):
    """
    Evaluate the performance of 5-fold cross validation.
    """
    res = read_prediction_results(result_dir)
    total_correct_equation = res["correct_equation"]
    total_correct_answer = res["correct_answer"]
    total_problems = res["total"]

    print(f"({total_correct_equation}, {total_correct_answer}, {total_problems})"
          f"\t {100* total_correct_answer/total_problems:.2f}% ({total_correct_answer}/{total_problems})")


def evaluate_performance_5fold(result_dir_5fold):
    """
    Evaluate a single directory.
    """
    total_correct_equation = 0
    total_correct_answer = 0
    total_problems = 0
    correct_answer_each_fold = []

    for i in range(5):
        result_dir = os.path.join(result_dir_5fold, f"fold{i}")
        res = read_prediction_results(result_dir)
        total_correct_equation += res["correct_equation"]
        total_correct_answer += res["correct_answer"]
        total_problems += res["total"]
        correct_answer_each_fold.append(res["correct_answer"])

    print(f"{', '.join([str(x) for x in correct_answer_each_fold])}")
    print(f"({total_correct_equation}, {total_correct_answer}, {total_problems})"
          f"\t {100* total_correct_answer/total_problems:.2f}% ({total_correct_answer}/{total_problems})")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir")
    parser.add_argument("--five_fold", action="store_true",
                        help="If the flag is on, this script will evaluate the five sub-directories of the input dir")
    args = parser.parse_args()

    if not os.path.isdir(args.result_dir):
        raise ValueError("Invalid input path!")
    elif args.five_fold is True:
        evaluate_performance_5fold(args.result_dir)
    else:
        evaluate_performance(args.result_dir)
