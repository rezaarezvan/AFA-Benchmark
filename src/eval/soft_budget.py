import logging

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from common.custom_types import (
    AFADataset,
    AFAPredictFn,
    AFASelectFn,
    AFAUncoverFn,
)

log = logging.getLogger(__name__)


def eval_soft_budget_afa_method(
    afa_select_fn: AFASelectFn,
    dataset: AFADataset,  # also assumed to subclass torch Dataset
    afa_uncover_fn: AFAUncoverFn,
    external_afa_predict_fn: AFAPredictFn | None = None,
    builtin_afa_predict_fn: AFAPredictFn | None = None,
    only_n_samples: int | None = None,
    device: torch.device | None = None,
    batch_size: int = 1,
    patch_size: int = 1,
) -> pd.DataFrame:
    """
    Evaluate an AFA method with support for early stopping and batched processing.

    Args:
        afa_select_fn (AFASelectFn): How to select new features. Should return 0 to stop.
        method_name (str): Name of the method, included in results DataFrame.
        dataset (AFADataset): The dataset to evaluate on. Additionally assumed to be a torch dataset.
        external_afa_predict_fn (AFAPredictFn): An external classifier.
        afa_uncover_fn (AFAUncoverFn): Function to that determines how to uncover features from AFA selections.
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

    acquisition_costs = dataset.get_feature_acquisition_costs().to(device)
    log.debug(f"Acquisition costs: {acquisition_costs}")

    # Optionally subset the dataset
    if only_n_samples is not None:
        dataset = torch.utils.data.Subset(dataset, range(only_n_samples))  # pyright: ignore[reportAssignmentType, reportArgumentType]

    dataloader = DataLoader(dataset, batch_size=batch_size)  # pyright: ignore[reportArgumentType]

    data_rows = []

    samples_to_eval = len(dataset)
    pbar = tqdm(
        total=samples_to_eval,
        desc="Evaluating dataset samples",
    )

    for _batch_features, _batch_label in dataloader:
        batch_features = _batch_features.to(device)
        if _batch_label.ndim == 2:
            batch_label = _batch_label.argmax(dim=1).to(device)
        elif _batch_label.ndim == 1:
            batch_label = _batch_label.to(device)
        else:
            raise ValueError(f"Unexpected label shape {_batch_label.shape}")

        # Initialize masks for the batch
        # TODO currently using 4D mask. Convert to 2D patches when evaluate on common classifier.
        batch_masked_features = torch.zeros_like(batch_features).to(device)
        batch_feature_mask = torch.zeros_like(
            batch_features, dtype=torch.bool
        ).to(device)

        # Keep track of which samples still need processing
        active_indices = torch.arange(batch_features.shape[0], device=device)

        while len(active_indices) > 0:
            log.debug(f"Active indices in batch: {active_indices}")

            # Only choose features for active samples
            active_batch_selection = afa_select_fn(
                batch_masked_features[active_indices],
                batch_feature_mask[active_indices],
                batch_features[active_indices],
                batch_label[active_indices],
            ).reshape(-1)
            assert active_batch_selection.ndim == 1, (
                f"batch_selection should be 1D, got {active_batch_selection.ndim}D with shape {active_batch_selection.shape}"
            )
            log.debug(f"Active batch selection: {active_batch_selection}")

            # Update masked features and feature mask
            _masked_features, _feature_mask = afa_uncover_fn(
                masked_features=batch_masked_features[active_indices],
                feature_mask=batch_feature_mask[active_indices],
                features=batch_features[active_indices],
                afa_selection=active_batch_selection,
            )
            batch_masked_features[active_indices] = _masked_features
            batch_feature_mask[active_indices] = _feature_mask

            # Check which active samples are now finished, either due to early stopping or observing all features
            just_finished_indices = active_indices[
                (active_batch_selection == 0)
                | (batch_feature_mask[active_indices].flatten(1).all(dim=-1))
            ]
            log.debug(
                f"Just finished indices in batch: {just_finished_indices}"
            )

            if len(just_finished_indices) > 0:
                # Run predictions for just finished samples
                log.debug(
                    f"Running predictions for just finished samples {just_finished_indices}"
                )
                if external_afa_predict_fn is not None:
                    external_prediction = external_afa_predict_fn(
                        batch_masked_features[just_finished_indices],
                        batch_feature_mask[just_finished_indices],
                        batch_features[just_finished_indices],
                        batch_label[just_finished_indices],
                    )
                    external_prediction = torch.argmax(
                        external_prediction, dim=-1
                    )
                else:
                    external_prediction = None

                if builtin_afa_predict_fn is not None:
                    builtin_prediction = builtin_afa_predict_fn(
                        batch_masked_features[just_finished_indices],
                        batch_feature_mask[just_finished_indices],
                        batch_features[just_finished_indices],
                        batch_label[just_finished_indices],
                    )
                    builtin_prediction = torch.argmax(
                        builtin_prediction, dim=-1
                    )
                else:
                    builtin_prediction = None

                for i, idx in enumerate(just_finished_indices):
                    fm = batch_feature_mask[idx]
                    if fm.dim() >= 3:
                        C, H, W = fm.shape[-3], fm.shape[-2], fm.shape[-1]
                        assert H % patch_size == 0 and W % patch_size == 0
                        ph, pw = H // patch_size, W // patch_size
                        patch_revealed = fm.view(
                            C, ph, patch_size, pw, patch_size
                        ).any(dim=(0, 2, 4))
                        patches_chosen = int(patch_revealed.sum().item())
                        features_chosen_val = patches_chosen
                        acquisition_cost_val = float(patches_chosen)
                    else:
                        features_chosen_val = int(fm.sum().item())
                        acquisition_cost_val = float(
                            (
                                acquisition_costs.flatten()
                                * fm.flatten().float()
                            )
                            .sum()
                            .item()
                        )

                    row = {
                        "features_chosen": features_chosen_val,
                        "predicted_label_external": None
                        if external_prediction is None
                        else external_prediction[i].item(),
                        # "true_label": batch_label[idx].argmax().item(),
                        "true_label": int(batch_label[idx].item()),
                        "predicted_label_builtin": None
                        if builtin_prediction is None
                        else builtin_prediction[i].item(),
                        "acquisition_cost": acquisition_cost_val,
                    }
                    data_rows.append(row)
                pbar.update(len(just_finished_indices))
                pbar.refresh()

                # Remove finished samples from active indices
                active_indices = active_indices[
                    ~torch.isin(active_indices, just_finished_indices)
                ]

    return pd.DataFrame(data_rows)
