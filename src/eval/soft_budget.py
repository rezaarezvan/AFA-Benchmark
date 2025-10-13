import logging

import pandas as pd
import torch
from tqdm import tqdm

from common.custom_types import (
    AFADataset,
    AFAPredictFn,
    AFASelectFn,
)

log = logging.getLogger(__name__)


def eval_soft_budget_afa_method(
    afa_select_fn: AFASelectFn,
    dataset: AFADataset,
    external_afa_predict_fn: AFAPredictFn,
    builtin_afa_predict_fn: AFAPredictFn | None = None,
    only_n_samples: int | None = None,
    device: torch.device | None = None,
    batch_size: int = 1,
) -> pd.DataFrame:
    """
    Evaluate an AFA method with support for early stopping and batched processing.

    Args:
        afa_select_fn (AFASelectFn): How to select new features. Should return 0 to stop.
        method_name (str): Name of the method, included in results DataFrame.
        dataset (AFADataset): The dataset to evaluate on. Additionally assumed to be a torch dataset.
        external_afa_predict_fn (AFAPredictFn): An external classifier.
        builtin_afa_predict_fn (AFAPredictFn): A builtin classifier, if such exists.
        only_n_samples (int|None, optional): If specified, only evaluate on this many samples from the dataset. Defaults to None.
        device (torch.device|None): Device to place data on. Defaults to "cpu".
        batch_size (int): Batch size for processing samples. Defaults to 1.

    Returns:
        pd.DataFrame: DataFrame containing columns:
            - "features_chosen"
            - "predicted_label_builtin"
            - "predicted_label_external"
    """
    if device is None:
        device = torch.device("cpu")

    data_rows = []

    # Retrieve all data at once, in order to do batched computations
    all_features, all_labels = dataset.get_all_data()
    remaining_indices = torch.arange(len(dataset))

    # Prepare initial batch, update as samples are completed
    batch_features = all_features[remaining_indices[:batch_size]].to(device)
    batch_label = all_labels[remaining_indices[:batch_size]].to(device)
    batch_masked_features = torch.zeros_like(batch_features).to(device)
    batch_feature_mask = torch.zeros_like(batch_features, dtype=torch.bool).to(
        device
    )

    # Keep track of how many samples have been completed. Once an item is completed, it is replaced by a new item from all_data.
    completed_samples = 0

    # Loop over the dataset
    samples_to_eval = (
        len(dataset) if only_n_samples is None else only_n_samples
    )
    pbar = tqdm(
        total=samples_to_eval,
        desc="Evaluating dataset samples",
    )
    while True:
        # Let AFA method do batched selection of new features (or stop)
        batch_selection = afa_select_fn(
            batch_masked_features,
            batch_feature_mask,
            batch_features,
            batch_label,
        ).squeeze(-1)
        assert batch_selection.ndim == 1, (
            f"batch_selection should be 1D, got {batch_selection.ndim}D with shape {batch_selection.shape}"
        )
        # Update masked features and feature mask individually. This could probably be done more efficiently with some advanced indexing, but this is clearer.
        for i in range(batch_size):
            sel = batch_selection[i].item()
            # Note that `sel == 1` means to select the first feature, that's why we subtract 1 here
            if sel > 0:
                batch_masked_features[i, sel - 1] = batch_features[i, sel - 1]
                batch_feature_mask[i, sel - 1] = True

        # Check which samples are finished, either due to early stopping or observing all features
        finished_mask = (batch_selection == 0) | (
            batch_feature_mask.all(dim=-1)
        )
        assert finished_mask.ndim == 1, (
            f"finished_mask should be 1D, got {finished_mask.ndim}D with shape {finished_mask.shape}"
        )
        finished_indices = torch.where(finished_mask)[0]

        # For finished samples, do predictions and store results
        external_prediction = external_afa_predict_fn(
            batch_masked_features[finished_mask],
            batch_feature_mask[finished_mask],
            batch_features[finished_mask],
            batch_label[finished_mask],
        )
        external_prediction = torch.argmax(external_prediction, dim=-1)

        # Builtin prediction, if available
        if builtin_afa_predict_fn is not None:
            builtin_prediction = builtin_afa_predict_fn(
                batch_masked_features[finished_mask],
                batch_feature_mask[finished_mask],
                batch_features[finished_mask],
                batch_label[finished_mask],
            )
            builtin_prediction = torch.argmax(builtin_prediction, dim=-1)
        else:
            builtin_prediction = None

        # For each finished sample, add a row to the DataFrame
        # "Local" index refers to the index within the finished sample, while "global" index refers to the index within the entire batch
        for finish_local_idx, finish_global_idx in enumerate(finished_indices):
            data_rows.append(
                {
                    "features_chosen": batch_feature_mask[finish_global_idx]
                    .sum()
                    .item(),
                    "predicted_label_builtin": builtin_prediction[
                        finish_local_idx
                    ],
                    "predicted_label_external": external_prediction[
                        finish_local_idx
                    ],
                    "true_label": batch_label[finish_global_idx]
                    .argmax()
                    .item(),
                }
            )

        n_finished = int(finished_mask.sum().item())
        completed_samples += n_finished
        pbar.update(n_finished)
        if completed_samples >= samples_to_eval:
            break
        if n_finished > 0:
            # Replace the finished samples with new samples from all_data
            n_to_refill = min(
                len(remaining_indices) - completed_samples,
                len(finished_indices),
            )
            new_indices = remaining_indices[
                completed_samples : completed_samples + n_to_refill
            ]
            refill_batch_indices = finished_indices[:n_to_refill]

            batch_features[refill_batch_indices] = all_features[
                new_indices
            ].to(device)
            batch_label[refill_batch_indices] = all_labels[new_indices].to(
                device
            )
            batch_masked_features[refill_batch_indices] = 0
            batch_feature_mask[refill_batch_indices] = False

    df = pd.DataFrame(data_rows)

    return df
