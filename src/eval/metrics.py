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
    budget: int,
    external_afa_predict_fn: AFAPredictFn,
    builtin_afa_predict_fn: AFAPredictFn | None = None,
    only_n_samples: int | None = None,
    device: torch.device | None = None,
) -> pd.DataFrame:
    """
    Evaluate an AFA method with support for early stopping and batched processing.

    Args:
        afa_select_fn (AFASelectFn): How to select new features. Should return 0 to stop.
        method_name (str): Name of the method, included in results DataFrame.
        dataset (AFADataset): The dataset to evaluate on. Additionally assumed to be a torch dataset.
        budget (int): The maximum number of features to select.
        external_afa_predict_fn (AFAPredictFn): An external classifier.
        builtin_afa_predict_fn (AFAPredictFn): A builtin classifier, if such exists.
        only_n_samples (int|None, optional): If specified, only evaluate on this many samples from the dataset. Defaults to None.
        device (torch.device|None): Device to place data on. Defaults to "cpu".

    Returns:
        pd.DataFrame: DataFrame containing columns:
            - "sample"
            - "features_chosen"
            - "predicted_label_builtin"
            - "predicted_label_external"
    """
    if only_n_samples is not None and only_n_samples:
        assert only_n_samples <= budget
        feature_limit = only_n_samples
    else:
        feature_limit = budget

    if device is None:
        device = torch.device("cpu")

    data_rows = []

    # Loop over the dataset
    for sample_idx, (_features, _label) in tqdm(  # pyright: ignore[reportGeneralTypeIssues]
        enumerate(dataset),  # pyright: ignore[reportArgumentType]
        desc="Evaluating dataset samples",
        total=len(dataset) if only_n_samples is None else only_n_samples,
    ):
        # Place data on device
        features = _features.to(device)
        label = _label.to(device)

        # Keep track of feature masks, update them as we choose more features
        masked_features = torch.zeros_like(features)
        feature_mask = torch.zeros_like(features, dtype=torch.bool)

        # Loop over hard budget
        for n_features_collected in range(feature_limit):
            # Let AFA method select a feature
            selection = int(
                afa_select_fn(
                    masked_features.unsqueeze(0),
                    feature_mask.unsqueeze(0),
                    features.unsqueeze(0),
                    label.unsqueeze(0),
                ).item()
            )
            if selection == 0:  # Stop acquiring features
                break
            else:
                # Update feature mask
                masked_features[selection - 1] = features[selection - 1]
                feature_mask[selection - 1] = True

        # Feature selection is done, either due to early stopping or reaching the hard budget

        # Make prediction with current features
        external_prediction = external_afa_predict_fn(
            masked_features.unsqueeze(0),
            feature_mask.unsqueeze(0),
            features.unsqueeze(0),
            label.unsqueeze(0),
        ).squeeze(0)
        external_prediction = int(external_prediction.argmax(dim=-1).item())

        # Builtin prediction, if available
        if builtin_afa_predict_fn is not None:
            builtin_prediction = builtin_afa_predict_fn(
                masked_features.unsqueeze(0),
                feature_mask.unsqueeze(0),
                features.unsqueeze(0),
                label.unsqueeze(0),
            ).squeeze(0)
            builtin_prediction = int(torch.argmax(builtin_prediction).item())
        else:
            builtin_prediction = None

        data_rows.append(
            {
                "sample": sample_idx,
                "features_chosen": n_features_collected + 1,
                "predicted_label_builtin": builtin_prediction,
                "predicted_label_external": external_prediction,
            }
        )

    df = pd.DataFrame(data_rows)

    return df
