import logging
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from common.custom_types import (
    AFADataset,
    AFAPredictFn,
    AFASelectFn,
)

log = logging.getLogger(__name__)


def aggregate_metrics(
    prediction_history: list[list[Tensor]],
    y_true: Tensor,
    actual_steps: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Compute accuracy, F1 and BCE across feature-selection budgets.

    If y_true contains exactly two unique classes   → average="binary"
    Otherwise                                       → average="weighted"

    Parameters
    ----------
    prediction_history : list[list[Tensor[float, shape=(n_classes,)]]]
        Variable-length prediction history for each sample
    y_true : Tensor[int, shape=(n_samples,)]
        Ground-truth labels, same order as prediction_history.
    actual_steps : Tensor[int, shape=(n_samples,)]
        Number of actual steps taken for each sample

    Returns
    -------
    accuracy_all : Tensor[float, shape=(max_steps,)]
    f1_all       : Tensor[float, shape=(max_steps,)]
    bce_all      : Tensor[float, shape=(max_steps,)]

    """
    max_steps = int(actual_steps.max())

    # Decide the F1 averaging mode
    classes = np.unique(y_true.cpu().numpy())
    if len(classes) == 2:
        f1_average = "binary"
        f1_pos_label = int(classes.max())
    else:
        f1_average = "weighted"
        f1_pos_label = None

    accuracy_all, f1_all, bce_all = [], [], []

    for step in range(max_steps):
        # Collect predictions for samples that have this step
        valid_predictions = []
        valid_labels = []

        for sample_idx, sample_steps in enumerate(actual_steps):
            if step < sample_steps:
                valid_predictions.append(prediction_history[sample_idx][step])
                valid_labels.append(y_true[sample_idx])

        if not valid_predictions:
            # No samples have this step, use NaN
            accuracy_all.append(float("nan"))
            f1_all.append(float("nan"))
            bce_all.append(float("nan"))
            continue

        valid_predictions = torch.stack(valid_predictions)
        valid_labels = torch.tensor(valid_labels)

        # Compute metrics for this step
        preds_i = torch.argmax(valid_predictions, dim=1)
        accuracy_all.append(
            accuracy_score(valid_labels.cpu().numpy(), preds_i.cpu().numpy())
        )
        if f1_pos_label is not None:
            f1_all.append(
                f1_score(
                    valid_labels.cpu().numpy(),
                    preds_i.cpu().numpy(),
                    average=f1_average,
                    pos_label=f1_pos_label,
                )
            )
        else:
            f1_all.append(
                f1_score(
                    valid_labels.cpu().numpy(),
                    preds_i.cpu().numpy(),
                    average=f1_average,
                )
            )

        # BCE requires one-hot encoding
        bce_all.append(
            F.binary_cross_entropy(
                valid_predictions,
                F.one_hot(
                    valid_labels, num_classes=valid_predictions.shape[-1]
                ).float(),
            ).item()
        )

    return (
        torch.tensor(accuracy_all, dtype=torch.float64),
        torch.tensor(f1_all, dtype=torch.float64),
        torch.tensor(bce_all, dtype=torch.float64),
    )


# def evaluator(
#     feature_mask_history_all: list[list[FeatureMask]],
#     prediction_history_all: list[list[Label]],
#     labels_all: list[Label],
# ) -> dict[str, Any]:
#     assert (
#         len(feature_mask_history_all) == len(prediction_history_all) == len(labels_all)
#     ), "All three lists must have the same length"
#
#     labels_all: Tensor = torch.stack(labels_all)
#     labels_all = torch.argmax(labels_all, dim=1)
#
#     accuracy_all, f1_all, bce_all = aggregate_metrics(
#         prediction_history_all, labels_all
#     )
#
#     return {
#         "accuracy_all": accuracy_all.detach().cpu(),
#         "f1_all": f1_all.detach().cpu(),
#         "bce_all": bce_all.detach().cpu(),
#         "feature_mask_history_all": [
#             [t.detach().cpu() for t in sublist] for sublist in feature_mask_history_all
#         ],
#     }


def _initialize_batch_state(
    features: Tensor, device: torch.device
) -> tuple[Tensor, Tensor, Tensor]:
    """Initialize the feature mask and masked features for a batch."""
    feature_mask = torch.zeros_like(features, dtype=torch.bool, device=device)
    masked_features = features.clone()
    masked_features[~feature_mask] = 0.0
    current_batch_size = features.shape[0]
    active_samples = torch.ones(
        current_batch_size, dtype=torch.bool, device=device
    )
    return feature_mask, masked_features, active_samples


def _process_selections(
    selections: Tensor,
    active_samples: Tensor,
    feature_mask: Tensor,
    masked_features: Tensor,
    features: Tensor,
    batch_feature_mask_histories: list[list[Tensor]],
) -> list[int]:
    """Process feature selections and update masks. Returns indices of samples to deactivate."""
    active_indices = torch.where(active_samples)[0]
    samples_to_deactivate = []

    for i, selection in enumerate(selections):
        original_idx = active_indices[i].item()

        if selection.item() == 0:
            # Sample wants to stop
            samples_to_deactivate.append(original_idx)
        elif selection.item() > 0:  # Valid feature selection
            # Convert from 1-based to 0-based indexing
            feature_idx = int(selection.item()) - 1
            orig_idx = int(original_idx)

            # Update feature mask and masked features
            feature_mask[orig_idx, feature_idx] = True
            masked_features[orig_idx, feature_idx] = features[
                orig_idx, feature_idx
            ]

            # Store feature mask state for this sample
            batch_feature_mask_histories[orig_idx].append(
                feature_mask[orig_idx].clone()
            )

    return samples_to_deactivate


def _get_active_batch_data(
    features: Tensor,
    labels: Tensor,
    feature_mask: Tensor,
    masked_features: Tensor,
    active_samples: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Extract data for only the active samples in the batch."""
    active_masked_features = masked_features[active_samples]
    active_feature_mask = feature_mask[active_samples]
    active_features = features[active_samples]
    active_labels = labels[active_samples]
    return (
        active_masked_features,
        active_feature_mask,
        active_features,
        active_labels,
    )


def _update_prediction_histories(
    predictions: Tensor,
    batch_prediction_histories: list[list[Tensor]],
    batch_feature_mask_histories: list[list[Tensor]],
    current_batch_size: int,
) -> None:
    """Update prediction histories for samples that have taken a step."""
    for i in range(current_batch_size):
        if len(batch_feature_mask_histories[i]) > len(
            batch_prediction_histories[i]
        ):
            batch_prediction_histories[i].append(predictions[i])


def _compute_action_distribution(
    all_feature_mask_histories: list[list[Tensor]], n_features: int
) -> Tensor:
    """Compute the distribution of features selected across all samples."""
    action_count = torch.zeros(n_features)

    for sample_history in all_feature_mask_histories:
        if sample_history:  # If sample took at least one step
            final_mask = sample_history[-1]
            action_count += final_mask.float().cpu()

    return (
        action_count / action_count.sum()
        if action_count.sum() > 0
        else action_count
    )


def _evaluate_batch(
    features: Tensor,
    labels: Tensor,
    afa_select_fn: AFASelectFn,
    afa_predict_fn: AFAPredictFn,
    budget: int,
    device: torch.device,
) -> tuple[list[list[Tensor]], list[list[Tensor]], int]:
    """
    Evaluate a single batch of samples.

    Returns:
        batch_prediction_histories: Prediction history for each sample
        batch_feature_mask_histories: Feature mask history for each sample
        current_batch_size: Number of samples in batch

    """
    current_batch_size = features.shape[0]

    # Initialize batch state
    feature_mask, masked_features, active_samples = _initialize_batch_state(
        features, device
    )

    # Store histories for each sample in batch
    batch_prediction_histories = [[] for _ in range(current_batch_size)]
    batch_feature_mask_histories = [[] for _ in range(current_batch_size)]

    # Process steps until all samples stop or budget reached
    for _ in range(budget):
        if not active_samples.any():
            break  # All samples have stopped

        # Get data for active samples only
        (
            active_masked_features,
            active_feature_mask,
            active_features,
            active_labels,
        ) = _get_active_batch_data(
            features, labels, feature_mask, masked_features, active_samples
        )

        if active_masked_features.shape[0] == 0:
            break  # No active samples

        # Get selections from method
        selections = afa_select_fn(
            active_masked_features,
            active_feature_mask,
            active_features,
            active_labels,
        ).squeeze(-1)

        # Process selections and get samples to deactivate
        samples_to_deactivate = _process_selections(
            selections,
            active_samples,
            feature_mask,
            masked_features,
            features,
            batch_feature_mask_histories,
        )

        # Deactivate samples that want to stop
        for idx in samples_to_deactivate:
            active_samples[idx] = False

        # Get predictions for all samples
        predictions = afa_predict_fn(
            masked_features, feature_mask, features, labels
        )

        # Update prediction histories
        _update_prediction_histories(
            predictions,
            batch_prediction_histories,
            batch_feature_mask_histories,
            current_batch_size,
        )

    return (
        batch_prediction_histories,
        batch_feature_mask_histories,
        current_batch_size,
    )


def eval_afa_method(
    afa_select_fn: AFASelectFn,
    dataset: AFADataset,  # NOTE: should also be annotated with torch.utils.data.Dataset
    budget: int,
    afa_predict_fn: AFAPredictFn,
    only_n_samples: int | None = None,
    batch_size: int = 1,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """
    Evaluate an AFA method with support for early stopping and batched processing.

    Args:
        afa_select_fn (AFASelectFn): How to select new features. Should return 0 to stop.
        dataset (AFADataset): The dataset to evaluate on. Additionally assumed to be a torch dataset.
        budget (int): The maximum number of features to select.
        afa_predict_fn (AFAPredictFn): The label prediction function to use for evaluation.
        only_n_samples (int|None, optional): If specified, only evaluate on this many samples from the dataset. Defaults to None.
        device (torch.device|None): Device to place data on. Defaults to "cpu".
        batch_size (int): How many AFA episodes to run concurrently.

    Returns:
        dict[str, float]: A dictionary containing the evaluation results.

    """
    if device is None:
        device = torch.device("cpu")

    # Store results for all samples
    all_prediction_histories = []  # list[list[Tensor[n_classes]]] - one per sample
    all_feature_mask_histories = []  # list[list[Tensor[n_features]]] - one per sample
    all_labels = []  # list[Tensor[n_classes]] - to be stacked into tensor
    all_actual_steps = []  # list[int] - number of steps taken per sample

    # Type ignore needed because AFADataset protocol doesn't inherit from Dataset
    # but implements the required methods
    dataloader = DataLoader(
        dataset,  # pyright: ignore[reportArgumentType]
        batch_size=batch_size,
        shuffle=False,
    )

    # Loop over the dataset
    n_evaluated_samples = 0
    for batch in tqdm(
        dataloader,
        total=only_n_samples // batch_size
        if only_n_samples is not None
        else len(dataloader),
        desc="Evaluating",
    ):
        # Each batch has features and labels
        features, labels = batch
        features = features.to(device)
        labels = labels.to(device)

        # Store true labels for this batch
        all_labels.append(labels)

        # Evaluate this batch
        (
            batch_prediction_histories,
            batch_feature_mask_histories,
            current_batch_size,
        ) = _evaluate_batch(
            features, labels, afa_select_fn, afa_predict_fn, budget, device
        )

        # Store results for this batch
        for i in range(current_batch_size):
            all_prediction_histories.append(batch_prediction_histories[i])
            all_feature_mask_histories.append(batch_feature_mask_histories[i])
            all_actual_steps.append(len(batch_prediction_histories[i]))

        n_evaluated_samples += current_batch_size
        if (
            only_n_samples is not None
            and n_evaluated_samples >= only_n_samples
        ):
            break

    # Convert to tensors for metrics computation
    labels_tensor = torch.cat(all_labels)
    labels_tensor = torch.argmax(labels_tensor, dim=1)
    actual_steps_tensor = torch.tensor(all_actual_steps)

    # Compute action distribution from final feature masks
    action_distribution = _compute_action_distribution(
        all_feature_mask_histories, dataset.n_features
    )

    log.info("Aggregating metrics...")
    accuracy_all, f1_all, bce_all = aggregate_metrics(
        all_prediction_histories, labels_tensor.cpu(), actual_steps_tensor
    )
    log.info("Finished aggregating metrics")

    return {
        "accuracy_all": accuracy_all,
        "f1_all": f1_all,
        "bce_all": bce_all,
        "action_distribution": action_distribution,
        "actual_steps": actual_steps_tensor,
        "average_steps": actual_steps_tensor.float().mean(),
    }
