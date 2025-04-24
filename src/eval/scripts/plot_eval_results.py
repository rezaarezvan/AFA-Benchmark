"""
Produce plots from evaluation results listed in EVALUATION_REGISTRY

One plot per training dataset.
One line per AFA method name.
Avg. and std. over validation datasets.
"""

import os
from pathlib import Path
import torch
from wandb.util import np
from common.registry import EVALUATION_REGISTRY, TRAINING_REGISTRY
import matplotlib.pyplot as plt


def get_result_mapping():
    """
    Produces a mapping like the following:
    {
        "cube_train_100.pt": {
            "random_dummy": [
                ("data/cube/cube_val_100.pt", "results/evaluation/random_dummy-cube_train_10000-cube_val_100.pt"),
                ("data/cube/cube_val_1000.pt", "results/evaluation/random_dummy-cube_train_10000-cube_val_1000.pt"),
            ],
            "sequential_dummy": ...
        }
    }
    """

    result_mapping = {}

    for (
        afa_method_path,
        val_data_path,
    ), eval_results_path in EVALUATION_REGISTRY.items():
        # Look up which AFA method and training dataset produced this afa_method_path
        temp = [k for k, v in TRAINING_REGISTRY.items() if v == afa_method_path]
        assert len(temp) == 1
        afa_method_name, train_data_path = temp[0]

        if train_data_path not in result_mapping:
            result_mapping[train_data_path] = {}
        if afa_method_name not in result_mapping[train_data_path]:
            result_mapping[train_data_path][afa_method_name] = []
        result_mapping[train_data_path][afa_method_name].append(
            (val_data_path, eval_results_path)
        )

    return result_mapping

def main():
    result_mapping = get_result_mapping()

    for train_data_path, methods in result_mapping.items():
        train_data_stem = Path(train_data_path).stem
        fig, ax = plt.subplots()
        for afa_method_name, eval_pairs in methods.items():
            #print(eval_pairs)
            accuracy = np.zeros(len(eval_pairs))
            avg_features_selected = np.zeros(len(eval_pairs))
            for i, (val_data_path, eval_results_path) in enumerate(eval_pairs):
                # Load the evaluation results
                eval_results = torch.load(eval_results_path)
                print(i)
                print(eval_results_path)
                #print(eval_results["accuracy_all"])

                # Extract the metrics
                accuracy[i] = eval_results["accuracy"]
                avg_features_selected[i] = eval_results["avg_features_selected"]
            print("@@@@")

            # Plot, one color per AFA method
            ax.plot(avg_features_selected, accuracy, label=afa_method_name)

        ax.set_title(f"Training dataset: {train_data_stem}")
        ax.set_xlabel("Avg. features selected")
        ax.set_ylabel("Accuracy")
        ax.legend()

        # Create folder if it doesn't exist
        os.makedirs("results/plots", exist_ok=True)
        plt.savefig(f"results/plots/{train_data_stem}.png")
        plt.close(fig)
        print(f"Plot saved for training dataset: {train_data_stem}")


if __name__ == "__main__":
    main()
