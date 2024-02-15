import os
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import os

import argparse
import os
from src.causal_tracing.causal_tracing import group_results
from src.detection.detection import generate_datasets, train_and_save
from src.utils.general import set_seed_everywhere
from src.utils.io import read_json

from src.utils.logger import freeze_args


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--causal_traces_dir", type=str
    )
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--features_to_include", nargs="+", required=False)
    parser.add_argument("--kinds_to_include", nargs="+", required=False)
    parser.add_argument("--n_samples_per_label", type=int)
    parser.add_argument("--train_ratio", type=float)
    parser.add_argument("--ablation_only_clean", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--ablation_include_corrupted", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--seed", default=100, type=int)

    return parser.parse_args()


def train_detector(args):
    models = {
        "LogisticRegression": {
            "model": LogisticRegression(max_iter=1000),
            "param_grid": {"C": [0.001, 0.01, 0.1, 1, 10, 100, 1000]},
        },
        "DecisionTree": {
            "model": DecisionTreeClassifier(),
            "param_grid": {
                "max_depth": [None, 5, 10, 15, 20],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
            },
        },
        "XGBoost": {
            "model": xgb.XGBClassifier(
                use_label_encoder=False, eval_metric="logloss", device="cuda", importance_type="total_gain"
            ),
            "param_grid": {
                "learning_rate": [0.01, 0.05, 0.1],
                "n_estimators": [100, 200, 500],
                "max_depth": [3, 5, 7, 10],
                "subsample": [0.8, 0.9, 1.0],
            },
        },
    }

    buckets = ["grounded", "unfaithful"]
    buckets_paths = [
        os.path.join(args.causal_traces_dir, args.dataset_name, args.model_name, f"{bucket}.json") for bucket in buckets
    ]

    results = []
    for bucket, bucket_path in zip(buckets, buckets_paths):
        results.append(group_results(read_json(bucket_path), bucket))

    # If we are only including certain kinds, filter the kinds
    if args.kinds_to_include is not None:
        results = [
            (
                {kind: bucket_results[0][kind] for kind in bucket_results[0] if kind in args.kinds_to_include},
                bucket_results[1],
                bucket_results[2],
            )
            for bucket_results in results
        ]

    # If we are only including certain features, filter the features
    if args.features_to_include is not None:
        results = [
            (
                {
                    kind: {
                        feature: bucket_results[0][kind][feature]
                        for feature in bucket_results[0][kind]
                        if feature in args.features_to_include
                    }
                    for kind in bucket_results[0]
                },
                bucket_results[1],
                bucket_results[2],
            )
            for bucket_results in results
        ]

    # Generate the datasets
    train_data, test_data, feature_names = generate_datasets(
        results[0],
        results[1],
        n_samples_per_label=args.n_samples_per_label,
        train_ratio=args.train_ratio,
        ablation_only_clean=args.ablation_only_clean,
        ablation_include_corrupted=args.ablation_include_corrupted,
    )

    # Train the models and save the results
    train_and_save(models, train_data, test_data, feature_names, class_names=buckets, seed=args.seed)


def main():
    args = get_args()
    freeze_args(args)
    set_seed_everywhere(args.seed)
    train_detector(args)


if __name__ == "__main__":
    main()
