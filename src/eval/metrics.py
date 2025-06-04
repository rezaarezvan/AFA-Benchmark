from typing import Any

import torch
from torch import Tensor
from tqdm import tqdm

from common.custom_types import (
    AFASelectFn,
    AFAPredictFn,
    AFADataset,
    FeatureMask,
    Label,
)
from sklearn.metrics import accuracy_score, f1_score
import numpy as np


def aggregate_metrics(prediction_history_all, y_true) -> tuple[Tensor, Tensor]:
    """
    Compute accuracy and F1 across feature-selection budgets.

    If y_true contains exactly two unique classes   → average="binary"
    Otherwise                                       → average="weighted"

    Parameters
    ----------
    prediction_history_all : list[list[int | float]]
        Outer list: one entry per test sample.
        Inner list: predictions for that sample after 1 … B selected features.
    y_true : array-like
        Ground-truth labels, same order as prediction_history_all.

    Returns
    -------
    accuracy_all : Tensor[float64]
    f1_all       : Tensor[float64]
    """
    if not prediction_history_all:
        return torch.tensor([], dtype=torch.float64), torch.tensor(
            [], dtype=torch.float64
        )

    B = len(prediction_history_all[0])
    if any(len(row) != B for row in prediction_history_all):
        raise ValueError("All inner lists must have the same length (budget B).")

    # Decide the F1 averaging mode
    classes = np.unique(y_true)
    if len(classes) == 2:
        f1_kwargs = {"average": "binary"}  # treat the larger label as positive
    else:
        f1_kwargs = {"average": "weighted"}

    accuracy_all, f1_all = [], []
    for i in range(B):
        preds_i = [torch.argmax(row[i]) for row in prediction_history_all]
        accuracy_all.append(accuracy_score(y_true, preds_i))
        f1_all.append(f1_score(y_true, preds_i, **f1_kwargs))

    return torch.tensor(accuracy_all, dtype=torch.float64), torch.tensor(
        f1_all, dtype=torch.float64
    )


def evaluator(
    feature_mask_history_all: list[list[FeatureMask]],
    prediction_history_all: list[list[Label]],
    labels_all: list[Label],
) -> dict[str, Any]:
    assert (
        len(feature_mask_history_all) == len(prediction_history_all) == len(labels_all)
    ), "All three lists must have the same length"

    labels_all: Tensor = torch.stack(labels_all)
    labels_all = torch.argmax(labels_all, dim=1)

    accuracy_all, f1_all = aggregate_metrics(prediction_history_all, labels_all)

    return {
        "accuracy_all": accuracy_all.detach().cpu(),
        "f1_all": f1_all.detach().cpu(),
        "feature_mask_history_all": [
            [t.detach().cpu() for t in sublist] for sublist in feature_mask_history_all
        ],
    }


def eval_afa_method(
    afa_select_fn: AFASelectFn,
    dataset: AFADataset,
    budget: int,
    afa_predict_fn: AFAPredictFn,
    only_n_samples: int | None = None,
) -> dict[str, Any]:
    """Evaluate an AFA method.

    Args:
        afa_select_fn (AFASelectFn): How to select new features.
        dataset (AFADataset): The dataset to evaluate on.
        budget (int): The number of features to select.
        afa_predict_fn (AFAPredictFn): The label prediction function to use for evaluation.
        only_n_samples (int|None, optional): If specified, only evaluate on this many samples from the dataset. Defaults to None.

    Returns:
        dict[str, float]: A dictionary containing the evaluation results.

    """
    # Store feature mask history, label prediction history, and true label for each sample in the dataset
    feature_mask_history_all: list[list[FeatureMask]] = []
    prediction_history_all: list[list[Label]] = []
    labels_all: list[Label] = []

    # Loop over the dataset
    n_evaluated_samples = 0
    for data in tqdm(
        iter(dataset),
        total=only_n_samples if only_n_samples is not None else len(dataset),
        desc="Evaluating",
    ):
        # Each datasample has a vector of features and a class (label)
        features, label = data

        # Immediately store the true label for this sample
        labels_all.append(label)

        # We will keep a history of which features have been observed, in case its relevant for evaluation
        feature_mask_history: list[FeatureMask] = []

        # And also a history of predictions
        prediction_history: list[Label] = []

        # Start with all features unobserved
        feature_mask = torch.zeros_like(features, dtype=torch.bool)
        masked_features = features.clone()
        masked_features[~feature_mask] = 0.0

        # Let AFA method select features for a fixed number of steps
        for _ in range(budget):
            # Always calculate a prediction
            prediction = afa_predict_fn(
                masked_features.unsqueeze(0), feature_mask.unsqueeze(0)
            ).squeeze(0)

            prediction_history.append(prediction)

            # Select new features
            selection = afa_select_fn(
                masked_features.unsqueeze(0), feature_mask.unsqueeze(0)
            ).squeeze(0)

            # Update the feature mask and masked features
            feature_mask[selection] = True
            masked_features[selection] = features[selection]

            # Store a copy of the feature mask in history
            feature_mask_history.append(feature_mask.clone())

        # Add the feature mask history and prediction history of this sample to the overall history
        feature_mask_history_all.append(feature_mask_history)
        prediction_history_all.append(prediction_history)

        n_evaluated_samples += 1
        if only_n_samples is not None and n_evaluated_samples >= only_n_samples:
            break

    # Now we have a history of feature masks and predictions for each sample in the dataset
    eval_results = evaluator(
        feature_mask_history_all, prediction_history_all, labels_all
    )

    return eval_results
