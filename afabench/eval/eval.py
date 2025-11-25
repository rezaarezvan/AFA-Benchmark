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
    AFAUnmaskFn,
    FeatureMask,
    Features,
    Label,
    MaskedFeatures,
)

log = logging.getLogger(__name__)


def process_batch(
    afa_select_fn: AFASelectFn,
    afa_unmask_fn: AFAUnmaskFn,
    features: Features,
    initial_feature_mask: FeatureMask,
    initial_masked_features: MaskedFeatures,
    true_label: Label,
    external_afa_predict_fn: AFAPredictFn | None = None,
    builtin_afa_predict_fn: AFAPredictFn | None = None,
    selection_budget: int | None = None,
) -> pd.DataFrame:
    """
    Evaluate a single batch.

    Args:
        afa_select_fn (AFASelectFn): How to choose AFA actions. Should return 0 to stop.
        afa_unmask_fn (AFAUnmaskFn): How to select new features from AFA actions.
        features (Features): Features for the batch.
        initial_feature_mask (FeatureMask): Initial feature mask for the batch.
        initial_masked_features (MaskedFeatures): Initial masked features for the batch.
        true_label (torch.Tensor): True labels for the batch.
        external_afa_predict_fn (AFAPredictFn): An external classifier.
        builtin_afa_predict_fn (AFAPredictFn): A builtin classifier, if such exists.
        selection_budget (int|None): How many AFA selections to allow per sample. If None, allow unlimited selections. Defaults to None.

    Returns:
        pd.DataFrame: DataFrame containing columns:
            - "selections_performed" (int)
            - "features_observed" (int)
            - "builtin_predicted_label" (int|None)
            - "external_predicted_label" (int|None)
            - "true_label" (int)
    """
    masked_features = initial_masked_features
    feature_mask = initial_feature_mask

    # Make AFA selections
    afa_selection = afa_select_fn(
        masked_features=masked_features,
        feature_mask=feature_mask,
    )

    # Translate into which unmasked features using the unmasker
    new_feature_mask, masked_features = afa_unmask_fn(
        feature_mask=feature_mask,
        masked_features=masked_features,
        afa_selection=afa_selection,
        features=features,
    )

    # Allow classifiers to make predictions using new masked features
    if external_afa_predict_fn is not None:
        external_prediction = external_afa_predict_fn(
            masked_features=masked_features,
            feature_mask=new_feature_mask,
        )
        # Assumption: predictions are logits or probabilities for classes
        assert external_prediction.shape[-1] > 1, (
            "Expected external prediction to have class dimension"
        )
        external_predicted_class = torch.argmax(external_prediction, dim=-1)
    else:
        external_prediction_class = None

    if builtin_afa_predict_fn is not None:
        builtin_prediction = builtin_afa_predict_fn(
            masked_features=masked_features,
            feature_mask=new_feature_mask,
        )
        # Assumption: predictions are logits or probabilities for classes
        assert builtin_prediction.shape[-1] > 1, (
            "Expected builtin prediction to have class dimension"
        )
        builtin_predicted_class = torch.argmax(builtin_prediction, dim=-1)

    # # TODO currently using 4D mask. Convert to 2D patches when evaluate on common classifier.
    # batch_masked_features = torch.zeros_like(batch_features).to(device)
    # batch_feature_mask = torch.zeros_like(batch_features, dtype=torch.bool).to(
    #     device
    # )

    # # Keep track of which samples still need processing
    # active_indices = torch.arange(batch_features.shape[], device=device)

    # while len(active_indices) > 0:
    #     log.debug(f"Active indices in batch: {active_indices}")

    #     # Only choose features for active samples
    #     active_batch_selection = afa_select_fn(
    #         batch_masked_features[active_indices],
    #         batch_feature_mask[active_indices],
    #         batch_features[active_indices],
    #         batch_label[active_indices],
    #     ).reshape(-1)
    #     assert (
    #         active_batch_selection.ndim == 1
    #     ), f"batch_selection should be 1D, got {
    #         active_batch_selection.ndim
    #     }D with shape {active_batch_selection.shape}"
    #     log.debug(f"Active batch selection: {active_batch_selection}")

    #     # Update masked features and feature mask
    #     _masked_features, _feature_mask = afa_uncover_fn(
    #         masked_features=batch_masked_features[active_indices],
    #         feature_mask=batch_feature_mask[active_indices],
    #         features=batch_features[active_indices],
    #         afa_selection=active_batch_selection,
    #     )
    #     batch_masked_features[active_indices] = _masked_features
    #     batch_feature_mask[active_indices] = _feature_mask

    #     # Check which active samples are now finished, either due to early stopping or observing all features
    #     just_finished_indices = active_indices[
    #         (active_batch_selection == 0)
    #         | (batch_feature_mask[active_indices].flatten(1).all(dim=-1))
    #     ]
    #     log.debug(f"Just finished indices in batch: {just_finished_indices}")

    #     if len(just_finished_indices) > 0:
    #         # Run predictions for just finished samples
    #         log.debug(
    #             f"Running predictions for just finished samples {
    #                 just_finished_indices
    #             }"
    #         )
    #         if external_afa_predict_fn is not None:
    #             external_prediction = external_afa_predict_fn(
    #                 batch_masked_features[just_finished_indices],
    #                 batch_feature_mask[just_finished_indices],
    #                 batch_features[just_finished_indices],
    #                 batch_label[just_finished_indices],
    #             )
    #             external_prediction = torch.argmax(external_prediction, dim=-1)
    #         else:
    #             external_prediction = None

    #         if builtin_afa_predict_fn is not None:
    #             builtin_prediction = builtin_afa_predict_fn(
    #                 batch_masked_features[just_finished_indices],
    #                 batch_feature_mask[just_finished_indices],
    #                 batch_features[just_finished_indices],
    #                 batch_label[just_finished_indices],
    #             )
    #             builtin_prediction = torch.argmax(builtin_prediction, dim=-1)
    #         else:
    #             builtin_prediction = None

    #         for i, idx in enumerate(just_finished_indices):
    #             fm = batch_feature_mask[idx]
    #             if fm.dim() >= 3:
    #                 C, H, W = fm.shape[-3], fm.shape[-2], fm.shape[-1]
    #                 assert H % patch_size == 0 and W % patch_size == 0
    #                 ph, pw = H // patch_size, W // patch_size
    #                 patch_revealed = fm.view(
    #                     C, ph, patch_size, pw, patch_size
    #                 ).any(dim=(0, 2, 4))
    #                 patches_chosen = int(patch_revealed.sum().item())
    #                 features_chosen_val = patches_chosen
    #                 acquisition_cost_val = float(patches_chosen)
    #             else:
    #                 features_chosen_val = int(fm.sum().item())
    #                 acquisition_cost_val = float(
    #                     (acquisition_costs.flatten() * fm.flatten().float())
    #                     .sum()
    #                     .item()
    #                 )

    #             row = {
    #                 "features_chosen": features_chosen_val,
    #                 "predicted_label_external": None
    #                 if external_prediction is None
    #                 else external_prediction[i].item(),
    #                 # "true_label": batch_label[idx].argmax().item(),
    #                 "true_label": int(batch_label[idx].item()),
    #                 "predicted_label_builtin": None
    #                 if builtin_prediction is None
    #                 else builtin_prediction[i].item(),
    #                 "acquisition_cost": acquisition_cost_val,
    #             }
    #             data_rows.append(row)
    #         pbar.update(len(just_finished_indices))
    #         pbar.refresh()

    #         # Remove finished samples from active indices
    #         active_indices = active_indices[
    #             ~torch.isin(active_indices, just_finished_indices)
    #         ]


def eval_afa_method(
    afa_select_fn: AFASelectFn,
    afa_unmask_fn: AFAUnmaskFn,
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
            - "selections_performed" (int)
            - "features_observed" (int)
            - "builtin_predicted_label" (int|None)
            - "external_predicted_label" (int|None)
            - "true_label" (int)
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
        batch_initial_feature_mask, batch_initial_masked_features = (
            afa_initialize_fn(batch_features)
        )
        df_batches.append(
            process_batch(
                afa_select_fn=afa_select_fn,
                afa_unmask_fn=afa_unmask_fn,
                initial_feature_mask=batch_initial_feature_mask,
                initial_masked_features=batch_initial_masked_features,
                true_label=batch_label,
                external_afa_predict_fn=external_afa_predict_fn,
                builtin_afa_predict_fn=builtin_afa_predict_fn,
                selection_budget=selection_budget,
            )
        )
    # Concatenate all batch DataFrames
    return pd.concat(df_batches, ignore_index=True)
