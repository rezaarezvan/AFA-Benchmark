from pathlib import Path
from typing import Any

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from common.utils import get_folders_with_matching_params


def get_classifier_paths_trained_on_data(
    classifier_type: str,
    train_dataset_path: Path,
    classifier_folder: Path = Path("models/classifiers"),
) -> list[Path]:
    """Get Paths to all classifiers of a specific type trained on a specific dataset."""
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
    assert "bce_all" in metrics, "Metrics must contain 'bce_all'."

    budget = len(metrics["accuracy_all"])
    fig, axs = plt.subplots(1, 2)
    budgets = np.arange(1, budget + 1, 1)

    # Convert to numpy arrays and handle NaN values
    accuracy_all = np.array(metrics["accuracy_all"])
    f1_all = np.array(metrics["f1_all"])
    bce_all = np.array(metrics["bce_all"])

    # Find valid (non-NaN) indices
    valid_acc = ~np.isnan(accuracy_all)
    valid_f1 = ~np.isnan(f1_all)
    valid_bce = ~np.isnan(bce_all)

    axs[0].plot(
        budgets[valid_acc],
        accuracy_all[valid_acc],
        label="Accuracy",
        marker="o",
    )
    axs[0].plot(
        budgets[valid_f1],
        f1_all[valid_f1],
        label="F1 Score",
        marker="o",
    )
    axs[0].set_xlabel("Number of Selected Features (Budget)")
    axs[0].legend()
    axs[0].set_title("Classification Metrics")

    axs[1].plot(
        budgets[valid_bce],
        bce_all[valid_bce],
        label="Binary Cross-Entropy",
        marker="o",
        color="red",
    )
    axs[1].set_xlabel("Number of Selected Features (Budget)")
    axs[1].set_ylabel("Binary Cross-Entropy")
    axs[1].set_title("Loss Metric")

    plt.tight_layout()
    return fig
