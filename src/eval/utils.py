from pathlib import Path
from typing import Any
from matplotlib.figure import Figure
import torch
from matplotlib import pyplot as plt
import numpy as np

from common.utils import get_folders_with_matching_params

from pathlib import Path
from typing import Any

import torch


def get_eval_results_with_fixed_keys(
    fixed_params_mapping: dict[str, Any] = {}, results_path=Path("results")
) -> list[dict[str, Any]]:
    """
    Return all evaluation results (as dictionaries) that have specific params values.
    The remaining keys are allowed to take any value.

    Args:
        fixed_params_mapping (dict[str, Any]): A dictionary mapping parameter names to their fixed values. Applies to the `params.yml` file in the results folder.
        results_path (Path): The path to the results folder. Defaults to "results".
    """

    return [
        torch.load(folder / "results.pt")
        for folder in get_folders_with_matching_params(
            results_path, fixed_params_mapping
        )
    ]


def get_classifier_paths_trained_on_data(
    classifier_type: str,
    train_dataset_path: Path,
    classifier_folder=Path("models/classifiers"),
) -> list[Path]:
    """
    Get Paths to all classifiers of a specific type trained on a specific dataset.
    """

    # Define the fixed parameters to match
    fixed_params_mapping = {"train_dataset_path": str(train_dataset_path)}

    # Get all matching folders
    matching_folders = get_folders_with_matching_params(
        classifier_folder / classifier_type, fixed_params_mapping
    )

    return matching_folders


def plot_metrics(metrics: dict[str, Any]) -> Figure:
    """Return a figure containing metrics."""
    assert "accuracy_all" in metrics, "Metrics must contain 'accuracy_all'."
    assert "f1_all" in metrics, "Metrics must contain 'f1_all'."

    budget = len(metrics["accuracy_all"])
    fig, ax = plt.subplots()
    budgets = np.arange(1, budget + 1, 1)
    ax.plot(
        budgets,
        metrics["accuracy_all"],
        label="Accuracy",
        marker="o",
    )
    ax.plot(
        budgets,
        metrics["f1_all"],
        label="F1 Score",
        marker="o",
    )
    ax.set_xlabel("Number of Selected Features (Budget)")
    return fig
