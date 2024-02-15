import os
import sys

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)

from scipy import stats
import argparse
from src.utils.io import read_json
from src.causal_tracing.causal_tracing import group_results

import numpy as np
from matplotlib import pyplot as plt


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--causal_traces_dir", type=str
    )
    parser.add_argument("--output_dir", type=str)

    return parser.parse_args()


def plot(dataset_name, model_name, grounded_results, unfaithful_results, save_path):
    titles = {
        "hidden": "Hidden activations",
        "mlp": "MLPs",
        "attn": "Attention heads"
    }
    model_names_conversion = {
        "llama2": "Llama2-7B",
        "llama": "LLaMA-7B",
        "gpt2": "GPT2-XL",
    }
    dataset_names_conversion = {
        "base_fakepedia": "Fakepedia-base",
        "multihop_fakepedia": "FakePedia-MH"
    }

    labels = [feature.get_name() for feature in next(iter(grounded_results[0].values())).values()]
    width = 0.8
    x = np.arange(len(labels))
    colors = {"grounded": "#FFC75F", "ungrounded": "#5390D9"}
    z_score = 1.96
    error_bar_props = {"capsize": 5, "capthick": 2, "elinewidth": 2}

    plt.rcParams.update({
        "font.size": 24,
        "font.family": "serif",
    })

    # Make a subplot for each bucket
    fig, axs = plt.subplots(1, 3, figsize=(30, 8))

    for i, kind in enumerate(["hidden", "mlp", "attn"]):
        for j, (bucket, results) in enumerate([("grounded", grounded_results), ("ungrounded", unfaithful_results)]):
            ax = axs[i]

            effects, corrupted_probs, clean_probs = results

            # Plot the three kind bars for each token
            for t, label in enumerate(labels):
                bar = ax.bar(
                    x[t] + (width / 4) * (["grounded", "ungrounded"].index(bucket) * 2 - 1),
                    effects[kind][label].avg() * 100,
                    width / 2,
                    yerr=effects[kind][label].std() * z_score / np.sqrt(len(effects[kind][label])) * 100,
                    color=colors[bucket],
                    error_kw=error_bar_props,
                    label=bucket if t == 0 else "",  # Label only the first bar for legend
                )

            # Perform a statistical test (t-test) to compare grounded and ungrounded results
            p_values = [stats.ttest_ind(
                grounded_results[0][kind][label].to_array(),
                unfaithful_results[0][kind][label].to_array()
            ).pvalue for label in labels]

            # Color-code x-axis labels based on p-values
            label_colors = []
            for p_value in p_values:
                if p_value < 0.01:
                    label_colors.append('red')  # Significant difference
                else:
                    label_colors.append('black')  # Not significant

            # Set the ticks and labels
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=28)

            for xtick, color in zip(ax.get_xticklabels(), label_colors):
                xtick.set_color(color)

            # Set the limits for the y-axis to be the same for both subplots
            ax.set_ylim([0, 100])

            # Set title for each subplot
            ax.set_title(titles[kind], fontsize=35, pad=20)

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            for v in [20, 40, 60, 80]:
                ax.axhline(y=v, linestyle='--', color='gray', linewidth=1, alpha=0.5)

            # Add legend
            if i == 0:  # Add legend in the first subplot only
                ax.set_ylabel('MGCT effect', fontsize=25)

            if i == 1:
                handles_leg, labels_leg = [], []
                for label_leg, color_leg in colors.items():
                    handles_leg.append(plt.Rectangle((0, 0), width, width, color=color_leg))
                    labels_leg.append(label_leg)

                ax.legend(handles_leg, labels_leg, loc='upper center', ncol=2, frameon=False)

    fig.suptitle(f"{model_names_conversion[model_name]} ({dataset_names_conversion[dataset_name]})", fontsize=45)

    plt.tight_layout()
    plt.subplots_adjust(left=0.10, right=0.90)

    # Add number of grounded and ungrounded facts represented
    save_path = save_path.replace(
        ".pdf", f"_grounded={len(grounded_results[1])}_ungrounded={len(unfaithful_results[1])}.pdf"
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", format='pdf')
    plt.close()


def plot_main(args):
    # List all dir names (each dir is a dataset)
    dataset_names = os.listdir(args.causal_traces_dir)

    for dataset_name in dataset_names:
        if "unfiltered" in dataset_name:
            continue

        # List all models (each model is a dir)
        model_names = os.listdir(os.path.join(args.causal_traces_dir, dataset_name))

        for model_name in model_names:

            buckets = ["grounded", "unfaithful"]
            buckets_paths = [
                os.path.join(args.causal_traces_dir, dataset_name, model_name, f"{bucket}.json") for bucket in buckets
            ]

            if not all([os.path.exists(bucket_path) for bucket_path in buckets_paths]):
                continue

            results = []
            for bucket, bucket_path in zip(buckets, buckets_paths):
                results.append(group_results(read_json(bucket_path), bucket))

            plot_path = os.path.join(args.output_dir, dataset_name, f"{model_name}.pdf")
            os.makedirs(os.path.dirname(plot_path), exist_ok=True)

            plot(dataset_name, model_name, results[0], results[1], plot_path)


def main():
    plot_main(get_args())


if __name__ == "__main__":
    main()
