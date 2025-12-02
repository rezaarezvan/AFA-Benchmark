import logging

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm

from afabench.common.custom_types import (
    AFADataset,
    AFAInitializeFn,
    AFAPredictFn,
    AFASelectFn,
    AFASelection,
    AFAUnmaskFn,
    FeatureMask,
    Features,
    Label,
    MaskedFeatures,
    SelectionMask,
)

log = logging.getLogger(__name__)


def single_afa_step(
    features: Features,
    label: Label,
    masked_features: MaskedFeatures,
    feature_mask: FeatureMask,
    selection_mask: SelectionMask,
    afa_select_fn: AFASelectFn,
    afa_unmask_fn: AFAUnmaskFn,
    feature_shape: torch.Size | None = None,
    external_afa_predict_fn: AFAPredictFn | None = None,
    builtin_afa_predict_fn: AFAPredictFn | None = None,
) -> tuple[
    AFASelection, MaskedFeatures, FeatureMask, Label | None, Label | None
]:
    """
    Perform a single AFA step.

    Args:
        features (Features): True unmasked features, required by unmasker.
        label (Label): The true label, passed to all functions that may need it. Usually not used, since that would be a form of cheating, but we might want some objects to have access to it for benchmarking.
        masked_features (MaskedFeatures): Currently masked features.
        feature_mask (FeatureMask): Current feature mask.
        selection_mask (SelectionMask): Mask indicating which selections have already been performed.
        afa_select_fn (AFASelectFn): How to make AFA selections.
        afa_unmask_fn (AFAUnmaskFn): How to select new features from AFA selections.
        feature_shape (torch.Size|None): Shape of the features, required by some objects.
        external_afa_predict_fn (AFAPredictFn|None): An external classifier.
        builtin_afa_predict_fn (AFAPredictFn|None): A builtin classifier, if such exists.

    Returns:
        tuple[AFASelection, MaskedFeatures, FeatureMask, Label|None, Label|None]: Selection made, updated masked features, feature mask and predicted labels after the AFA step.
    """
    # Make AFA selections
    active_afa_selection = afa_select_fn(
        masked_features=masked_features,
        feature_mask=feature_mask,
        selection_mask=selection_mask,
        label=label,
        feature_shape=feature_shape,
    )

    # Translate into which unmasked features using the unmasker
    new_feature_mask = afa_unmask_fn(
        masked_features=masked_features,
        feature_mask=feature_mask,
        features=features,
        afa_selection=active_afa_selection,
        selection_mask=selection_mask,
        label=label,
        feature_shape=feature_shape,
    )
    new_masked_features = features.clone()
    new_masked_features[~new_feature_mask] = 0.0

    # Allow classifiers to make predictions using new masked features
    if external_afa_predict_fn is not None:
        external_prediction = external_afa_predict_fn(
            masked_features=new_masked_features,
            feature_mask=new_feature_mask,
            label=label,
            feature_shape=feature_shape,
        )
    else:
        external_prediction = None

    if builtin_afa_predict_fn is not None:
        builtin_prediction = builtin_afa_predict_fn(
            masked_features=new_masked_features,
            feature_mask=new_feature_mask,
            label=label,
            feature_shape=feature_shape,
        )
    else:
        builtin_prediction = None

    return (
        active_afa_selection,
        new_masked_features,
        new_feature_mask,
        external_prediction,
        builtin_prediction,
    )


def process_batch(
    afa_select_fn: AFASelectFn,
    afa_unmask_fn: AFAUnmaskFn,
    n_selection_choices: int,
    features: Features,
    initial_feature_mask: FeatureMask,
    initial_masked_features: MaskedFeatures,
    true_label: Label,
    feature_shape: torch.Size | None = None,
    external_afa_predict_fn: AFAPredictFn | None = None,
    builtin_afa_predict_fn: AFAPredictFn | None = None,
    selection_budget: int | None = None,
) -> pd.DataFrame:
    """
    Evaluate a single batch.

    Assumes that predictions are for classes, and only stores the most likely class prediction.

    Args:
        afa_select_fn (AFASelectFn): How to make AFA selections. Should return 0 to stop.
        afa_unmask_fn (AFAUnmaskFn): How to select new features from AFA selections.
        n_selection_choices (int): Number of possible AFA selections (excluding 0). Should reflect the AFAUnmaskFn behavior.
        features (Features): Features for the batch.
        initial_feature_mask (FeatureMask): Initial feature mask for the batch.
        initial_masked_features (MaskedFeatures): Initial masked features for the batch.
        true_label (torch.Tensor): True labels for the batch.
        feature_shape (torch.Size|None): Shape of the features, required by some objects.
        external_afa_predict_fn (AFAPredictFn): An external classifier.
        builtin_afa_predict_fn (AFAPredictFn): A builtin classifier, if such exists.
        selection_budget (int|None): How many AFA selections to allow per sample. If None, allow unlimited selections. Defaults to None.

    Returns:
        pd.DataFrame: DataFrame containing columns:
            - "feature_indices" (list[int])
            - "prev_selections_performed" (list[int])
            - "selection_performed" (int)
            - "next_feature_indices" (list[int])
            - "builtin_predicted_class" (int|None)
            - "external_predicted_class" (int|None)
            - "true_class" (int)
    """
    # TODO: remove cloning if necessary for speed up
    features = features.clone()
    feature_mask = initial_feature_mask.clone()
    masked_features = initial_masked_features.clone()
    selection_mask = torch.zeros(
        (features.shape[0], n_selection_choices),
        device=features.device,
        dtype=torch.bool,
    )

    # In order to include "prev_selections_performed" in the dataframe, we keep track of which selections have been made per sample, which is not necessarily the same as the features observed
    selections_performed = [[] for _ in range(features.shape[0])]

    # Process a subset of the batch, which gets smaller and smaller until it's empty
    active_indices = torch.arange(features.shape[0], device=features.device)

    df_batch_rows = []

    while len(active_indices) > 0:
        active_features = features[active_indices]
        active_masked_features = masked_features[active_indices]
        active_feature_mask = feature_mask[active_indices]
        active_selection_mask = selection_mask[active_indices]

        (
            active_afa_selection,
            active_new_masked_features,
            active_new_feature_mask,
            active_external_prediction,
            active_builtin_prediction,
        ) = single_afa_step(
            features=active_features,
            label=true_label,
            masked_features=active_masked_features,
            feature_mask=active_feature_mask,
            afa_select_fn=afa_select_fn,
            afa_unmask_fn=afa_unmask_fn,
            feature_shape=feature_shape,
            external_afa_predict_fn=external_afa_predict_fn,
            builtin_afa_predict_fn=builtin_afa_predict_fn,
            selection_mask=active_selection_mask,
        )
        # Key assumption: predictions are logits/probabilities for classes
        if active_builtin_prediction is not None:
            assert active_external_prediction.shape[-1] > 1, (
                "Expected external prediction to have class dimension"
            )
        if builtin_afa_predict_fn is not None:
            assert active_builtin_prediction.shape[-1] > 1, (
                "Expected builtin prediction to have class dimension"
            )

        # Append selections
        for global_active_idx, afa_selection in zip(
            active_indices, active_afa_selection, strict=True
        ):
            selections_performed[global_active_idx].append(
                afa_selection.item()
            )

        # Append one row per active sample
        for active_idx, true_idx in enumerate(active_indices):
            df_batch_rows.append(
                {
                    "feature_indices": active_feature_mask[active_idx]
                    .nonzero(as_tuple=False)
                    .flatten()
                    .cpu()
                    .tolist(),
                    "prev_selections_performed": selections_performed[
                        int(true_idx.item())
                    ][:-1],
                    "selection_performed": active_afa_selection[
                        active_idx
                    ].item(),
                    "next_feature_indices": active_new_feature_mask[active_idx]
                    .nonzero(as_tuple=False)
                    .flatten()
                    .cpu()
                    .tolist(),
                    "builtin_predicted_label": None
                    if active_builtin_prediction is None
                    else active_builtin_prediction[active_idx]
                    .argmax(-1)
                    .item(),
                    "external_predicted_label": None
                    if active_external_prediction is None
                    else active_external_prediction[active_idx]
                    .argmax(-1)
                    .item(),
                    "true_class": true_label[true_idx].argmax(-1).item(),
                }
            )

        # Check which active samples have finished, due to one of the following reasons:
        # - selection == 0 (method chose to stop)
        # - selection budget reached
        # - all features unmasked
        just_finished_mask = (
            active_afa_selection.squeeze(-1) == 0
        ) | active_new_feature_mask.flatten(start_dim=1).all(dim=1)
        # Check if selection budget is reached
        for active_idx, selection_list in enumerate(selections_performed):
            if len(selection_list) >= (selection_budget or float("inf")):
                just_finished_mask[active_idx] = True
        # Update feature mask, masked features and selection mask
        masked_features[active_indices] = active_new_masked_features
        feature_mask[active_indices] = active_new_feature_mask
        sel = active_afa_selection.squeeze(-1)
        valid = sel > 0
        if valid.any():
            selection_mask[active_indices[valid], sel[valid] - 1] = True

        # Filter out finished samples
        active_indices = active_indices[~just_finished_mask]

    return pd.DataFrame(df_batch_rows)


def eval_afa_method(
    afa_select_fn: AFASelectFn,
    afa_unmask_fn: AFAUnmaskFn,
    n_selection_choices: int,
    afa_initialize_fn: AFAInitializeFn,
    dataset: AFADataset,  # we also check at runtime that this is a pytorch Dataset
    external_afa_predict_fn: AFAPredictFn | None = None,
    builtin_afa_predict_fn: AFAPredictFn | None = None,
    only_n_samples: int | None = None,
    device: torch.device | None = None,
    selection_budget: int | None = None,
    batch_size: int = 1,
) -> pd.DataFrame:
    """
    Evaluate an AFA method with support for early stopping and batched processing.

    Args:
        afa_select_fn (AFASelectFn): How to choose AFA actions. Should return 0 to stop.
        afa_unmask_fn (AFAUnmaskFn): How to select new features from AFA actions.
        n_selection_choices (int): Number of possible AFA selections (excluding 0).
        afa_initialize_fn (AFAInitializeFn): How to create the initial feature mask.
        dataset (AFADataset & Dataset): The dataset to evaluate on. Additionally assumed to be a torch dataset.
        external_afa_predict_fn (AFAPredictFn): An external classifier.
        builtin_afa_predict_fn (AFAPredictFn): A builtin classifier, if such exists.
        only_n_samples (int|None, optional): If specified, only evaluate on this many samples from the dataset. Defaults to None.
        device (torch.device|None): Device to place data on. Defaults to "cpu".
        selection_budget (int|None): How many AFA selections to allow per sample. If None, allow unlimited selections. Defaults to None.
        batch_size (int): Batch size for processing samples. Defaults to 1.

    Returns:
        pd.DataFrame: DataFrame containing columns:
            - "feature_indices" (list[int])
            - "prev_selections_performed" (list[int])
            - "selection_performed" (int)
            - "next_feature_indices" (list[int])
            - "builtin_predicted_class" (int|None)
            - "external_predicted_class" (int|None)
            - "true_class" (int)
    """
    assert isinstance(dataset, Dataset)
    if device is None:
        device = torch.device("cpu")

    if only_n_samples is not None:
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=SubsetRandomSampler(
                (torch.randperm(len(dataset))[:only_n_samples]).tolist()
            ),
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
        )

    df_batches: list[pd.DataFrame] = []
    for _batch_features, _batch_label in tqdm(dataloader):
        batch_features = _batch_features.to(device)
        batch_label = _batch_label.to(device)

        # Initialize masks for the batch
        batch_initial_feature_mask = afa_initialize_fn(
            batch_features,
            batch_label,
            feature_shape=dataset.feature_shape,
        )
        batch_initial_masked_features = batch_features.clone()
        batch_initial_masked_features[~batch_initial_feature_mask] = (
            0.0  # Assuming zero masking
        )

        df_batches.append(
            process_batch(
                afa_select_fn=afa_select_fn,
                afa_unmask_fn=afa_unmask_fn,
                n_selection_choices=n_selection_choices,
                features=batch_features,
                initial_feature_mask=batch_initial_feature_mask,
                initial_masked_features=batch_initial_masked_features,
                true_label=batch_label,
                feature_shape=dataset.feature_shape,
                external_afa_predict_fn=external_afa_predict_fn,
                builtin_afa_predict_fn=builtin_afa_predict_fn,
                selection_budget=selection_budget,
            )
        )
    # Concatenate all batch DataFrames
    return pd.concat(df_batches, ignore_index=True)
