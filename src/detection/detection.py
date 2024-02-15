import json

import numpy as np
from typing import Tuple
import os
import json
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import json
import numpy as np
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import plot_tree
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

from src.utils.logger import get_logger, get_output_dir


def generate_datasets(
    grounded_results,
    unfaithful_results,
    train_ratio=0.8,
    n_samples_per_label=2000,
    ablation_only_clean=False,
    ablation_include_corrupted=False,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    logger = get_logger()
    buckets = [grounded_results, unfaithful_results]

    if ablation_only_clean:
        feature_names = [grounded_results[2].get_name()]

        if ablation_include_corrupted:
            feature_names.append(grounded_results[1].get_name())

    else:
        feature_names = [
            f"{kind}-{feature}" for kind, features in grounded_results[0].items() for feature in features.keys()
        ]

    logger.info(f"Feature names: {feature_names}")

    all_samples = []
    all_labels = []

    for label, bucket_results in enumerate(buckets):
        kinds_results, corr_probs, clean_probs = bucket_results

        num_samples = len(corr_probs)

        logger.info("Number of samples: {}".format(num_samples))

        current_label_samples = []

        for i in range(num_samples):
            if ablation_only_clean:
                candidate_example = [clean_probs.get(i)]

                if ablation_include_corrupted:
                    candidate_example.append(corr_probs.get(i))

            else:
                candidate_example = [
                    feature_results.get(i)
                    for kind_results in kinds_results.values()
                    for feature_results in kind_results.values()
                ]

                if any([feature is None for feature in candidate_example]):
                    continue

            current_label_samples.append((candidate_example, label))

        if len(current_label_samples) < n_samples_per_label:
            raise ValueError(
                f"Bucket {label} has fewer than {n_samples_per_label} valid samples! In particular, there are {len(current_label_samples)} samples."
            )

        # Shuffle the samples for this label and take the first n_samples_per_label samples
        np.random.shuffle(current_label_samples)
        all_samples.extend([sample[0] for sample in current_label_samples[:n_samples_per_label]])
        all_labels.extend([sample[1] for sample in current_label_samples[:n_samples_per_label]])

    # Convert all_samples and all_labels to np arrays
    all_samples_array = np.array(all_samples)
    all_labels_array = np.array(all_labels)

    # Calculate lengths for each split
    total_size = len(all_samples_array)
    train_size = int(total_size * train_ratio)

    # Shuffle and split the dataset
    indices = np.arange(total_size)
    np.random.shuffle(indices)

    train_dataset = (all_samples_array[indices[:train_size]], all_labels_array[indices[:train_size]])
    test_dataset = (all_samples_array[indices[train_size:]], all_labels_array[indices[train_size:]])
    return train_dataset, test_dataset, feature_names


def load_metrics(save_dir):
    with open(os.path.join(save_dir, "results.json"), "r") as file:
        results = json.load(file)
    return results


def save_metrics(results, feature_names, save_dir):
    with open(os.path.join(save_dir, "results.json"), "w") as file:
        json.dump(results, file, indent=4)

    if "feature_importances" in results:
        importances = results["feature_importances"]
        indices = np.argsort(importances)

        # Logic to determine the kind for colors
        colors = {"hidden": "grey", "mlp": "blue", "attn": "orange", "corr": "grey", "clean": "grey"}

        def determine_kind(verbose_name):
            for kind, color in colors.items():
                if kind in verbose_name.lower():
                    return color
            print(f"Unmatched feature: {verbose_name}")
            raise ValueError("Unknown feature kind.")

        bar_colors = [determine_kind(name) for name in feature_names]

        # Make the font size larger
        plt.rcParams.update({"font.size": 21})

        # Change the font family
        plt.rcParams["font.family"] = "serif"

        plt.figure(figsize=(15, 15))
        plt.barh(
            range(len(indices)),
            [importances[i] for i in indices],
            align="center",
            color=[bar_colors[i] for i in indices],
            edgecolor="white",
        )
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel("Relative Importance")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "feature_importances.png"))
        plt.close()


def save_decision_tree_plot(tree, feature_names, class_names, save_dir):
    plt.figure(figsize=(20, 10))
    plot_tree(tree, feature_names=feature_names, class_names=class_names, filled=True)
    plt.savefig(os.path.join(save_dir, "decision_tree.png"))
    plt.close()


def plot_metrics_comparison(metrics_by_model, save_dir):
    """
    metrics_by_model: dict, keys are model names (like 'Logistic Regression', 'DecisionTree', 'XGBoost') and values are
                      dictionaries of metrics (keys are metric names, values are metric values)
    save_dir: directory where plots will be saved
    """
    model_colors = {"LogisticRegression": "grey", "DecisionTree": "orange", "XGBoost": "blue"}

    # Validate that all models in metrics_by_model are known
    for model in metrics_by_model:
        if model not in model_colors:
            raise Exception(f"Unknown model: {model}")

    n_models = len(metrics_by_model)
    n_metrics = len(metrics_by_model[next(iter(metrics_by_model))])

    # Set bar width, distance between bars in a group, and positions
    bar_width = 0.2
    distance = 0.05  # distance between bars in a group
    r1 = np.arange(n_metrics)  # positions for first model
    r2 = [x + bar_width + distance for x in r1]  # positions for second model
    r3 = [x + bar_width + distance for x in r2]  # positions for third model

    # Make the font size larger
    plt.rcParams.update({"font.size": 21})

    # Change the font family
    plt.rcParams["font.family"] = "serif"

    plt.figure(figsize=(15, 10))

    # Plotting bars for each model
    all_metric_values = []
    for idx, (model, metrics) in enumerate(metrics_by_model.items()):
        metric_values = [metrics[metric] for metric in metrics]
        all_metric_values.extend(metric_values)
        positions = [r1, r2, r3][idx]
        plt.bar(positions, metric_values, color=model_colors[model], width=bar_width, edgecolor="white", label=model)

    # Adjust y-axis limit
    plt.ylim(bottom=min(all_metric_values) * 0.9)

    plt.xlabel("Metrics")
    plt.ylabel("Score")
    xtick_positions = [r2[i] for i in range(n_metrics)]  # Averages of r1 and r2 positions
    plt.xticks(xtick_positions, list(metrics_by_model[next(iter(metrics_by_model))]))

    # Place the legend outside the plot on the right
    plt.legend()
    plt.tight_layout()

    plt.savefig(os.path.join(save_dir, "all_metrics_comparison.png"), bbox_inches="tight")
    plt.close()


def train_and_save(models, train_data, test_data, feature_names, class_names, seed, replot_only=False):
    save_dir = get_output_dir()
    plt.rcParams["font.size"] = max(1, plt.rcParams["font.size"])

    metrics_by_model = {}

    for model_name, model_info in models.items():
        model_save_dir = os.path.join(save_dir, model_name)
        os.makedirs(model_save_dir, exist_ok=True)

        if not replot_only:
            X_train, y_train = train_data
            X_test, y_test = test_data

            if "random_state" in model_info["model"].get_params():
                model_info["model"].set_params(random_state=seed)

            clf = GridSearchCV(model_info["model"], model_info["param_grid"], cv=5, verbose=10)
            clf.fit(X_train, y_train)

            y_pred = clf.predict(X_test)
            y_train_pred = clf.predict(X_train)
            y_train_proba = clf.predict_proba(X_train)[:, 1]

            results = {
                "train": {
                    "accuracy": accuracy_score(y_train, y_train_pred),
                    "precision": precision_score(y_train, y_train_pred),
                    "recall": recall_score(y_train, y_train_pred),
                    "f1_score": f1_score(y_train, y_train_pred),
                    "roc_auc": roc_auc_score(y_train, y_train_proba),
                },
                "test": {
                    "accuracy": accuracy_score(y_test, y_pred),
                    "precision": precision_score(y_test, y_pred),
                    "recall": recall_score(y_test, y_pred),
                    "f1_score": f1_score(y_test, y_pred),
                    "roc_auc": roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]),
                },
                "best_hyperparameters": clf.best_params_,
            }

            if hasattr(clf.best_estimator_, "feature_importances_"):
                # If there is an importance type attribute, print it
                if hasattr(clf.best_estimator_, "importance_type"):
                    print(f"Feature importances: {clf.best_estimator_.importance_type}")
                results["feature_importances"] = list(clf.best_estimator_.feature_importances_)
                results["feature_importances"] = [float(val) for val in results["feature_importances"]]

            if isinstance(clf.best_estimator_, LogisticRegression):
                # Taking the absolute values of the coefficients
                results["feature_importances"] = [float(abs(val)) for val in clf.best_estimator_.coef_.flatten()]

            save_metrics(results, feature_names, model_save_dir)

            if model_name == "DecisionTree":
                save_decision_tree_plot(clf.best_estimator_, feature_names, class_names, model_save_dir)
        else:
            results = load_metrics(model_save_dir)
            save_metrics(results, feature_names, model_save_dir)

        metrics_by_model[model_name] = results["test"]

    plot_metrics_comparison(metrics_by_model, save_dir)
