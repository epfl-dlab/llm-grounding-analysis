import json
import os
import sys

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)

import argparse
from src.utils.io import read_json


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--descriptive_analysis_dir", type=str
    )
    parser.add_argument(
        "--output_dir", type=str
    )
    parser.add_argument("--strict_grounding", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def plot_main(args):

    results = {"filtered_base_fakepedia": {}}

    # List all dir names (each dir is a dataset)
    dataset_names = os.listdir(args.descriptive_analysis_dir)

    for dataset_name in dataset_names:
        results[dataset_name] = {}

        # List all models (each model is a file)
        model_files_paths = os.listdir(os.path.join(args.descriptive_analysis_dir, dataset_name))
        model_names = [".".join(model_file_path.split(".")[:-1]) for model_file_path in model_files_paths]

        # Sort the models by name
        model_files_paths, model_names = zip(*sorted(zip(model_files_paths, model_names), key=lambda x: x[1]))

        for model_name, model_file_path in zip(model_names, model_files_paths):
            model_file_path = os.path.join(args.descriptive_analysis_dir, dataset_name, model_file_path)
            model_results = read_json(model_file_path)

            temp = {
                "count_facts": 0.0,
                "count_grounded": 0.0,
            }

            for fact in model_results:

                # If any of the answer is None, then we skip this fact
                if (
                    fact["answers"]["option_a_grounded"]["answer"] is None
                    or fact["answers"]["option_b_grounded"]["answer"] is None
                ):
                    continue

                temp["count_facts"] += 1.0

                score_to_add = 0.0
                if (
                    fact["answers"]["option_a_grounded"]["grounded"]
                    and fact["answers"]["option_b_grounded"]["grounded"]
                ):
                    score_to_add = 1.0
                else:
                    if not args.strict_grounding and (
                        fact["answers"]["option_a_grounded"]["grounded"]
                        or fact["answers"]["option_b_grounded"]["grounded"]
                    ):
                        score_to_add = 0.5
                temp["count_grounded"] += score_to_add

            results[dataset_name][model_name] = temp

    # Compute the percentage of grounded facts for each model
    for dataset_name in results:
        for model_name in results[dataset_name]:
            results[dataset_name][model_name]["grounded_percentage"] = (
                results[dataset_name][model_name]["count_grounded"]
                / results[dataset_name][model_name]["count_facts"]
                * 100
            )

    print(json.dumps(results, indent=4))


def main():
    plot_main(get_args())


if __name__ == "__main__":
    main()
