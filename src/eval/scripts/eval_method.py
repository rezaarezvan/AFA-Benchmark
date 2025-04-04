"""
Evaluates an AFA method (as defined in common.custom_types) on a specific dataset.

Prints out the average accuracy achieved, and the average number of features selected.

"""

import argparse

import torch
from tqdm import tqdm

from common.custom_types import (
    AFADataset,
    AFAMethod,
    FeatureMask,
    Features,
    Label,
)
from eval.registry import AFA_DATASET_REGISTRY, AFA_METHOD_REGISTRY


def evaluator(feature_mask_history: list[FeatureMask], label: torch.Tensor) -> dict:
    """
    Decides which metrics to use for evaluation.
    """
    return {"accuracy": 0.42, "num_features_selected": 2, "time": 0.1}


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate an AFAMethod on a specific AFADataset"
    )
    parser.add_argument(
        "--afa_method_name",
        type=str,
        required=True,
        help="Name of the AFAMethod to evaluate. Must be one of: "
        + ", ".join(AFA_METHOD_REGISTRY.keys()),
    )
    parser.add_argument(
        "--afa_method_path",
        type=str,
        required=True,
        help="Path that will be passed to the AFAMethod's load method. ",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Name of the AFADataset to evaluate. Must be one of: "
        + ", ".join(AFA_DATASET_REGISTRY.keys()),
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to a .pt file containing the AFADataset to evaluate the AFAMethod on",
    )
    args = parser.parse_args()

    if args.afa_method_name not in AFA_METHOD_REGISTRY:
        raise ValueError(
            f"Method {args.afa_method_name} not in registry. Must be one of: "
            + ", ".join(AFA_METHOD_REGISTRY.keys())
        )
    if args.dataset_name not in AFA_DATASET_REGISTRY:
        raise ValueError(
            f"Method {args.afa_dataset_name} not in registry. Must be one of: "
            + ", ".join(AFA_DATASET_REGISTRY.keys())
        )

    # Load the AFA method
    afa_method: AFAMethod = AFA_METHOD_REGISTRY[args.afa_method_name](
        args.afa_method_path
    )
    print(f"Loaded AFA method {args.afa_method_name} from {args.afa_method_path}")

    # Load the dataset
    dataset: AFADataset = AFA_DATASET_REGISTRY[args.dataset_name](args.dataset_path)
    print(f"Loaded dataset {args.dataset_name} from {args.dataset_path}")

    # Several metrics per sample in the dataset
    all_metrics: list[dict] = []

    # Loop over the dataset
    for data in tqdm(iter(dataset), total=len(dataset), desc="Evaluating"):
        # Each datasample has a vector of features and a class (label)
        features, label = data

        # AFA methods expect a batch dimension
        features: Features = features.unsqueeze(0)
        label: Label = label.unsqueeze(0)

        # We will keep a history of which features have been observed, in case its relevant for evaluation
        feature_mask_history: list[FeatureMask] = []

        # Start with all features unobserved
        feature_mask: FeatureMask = torch.zeros_like(features, dtype=torch.bool)
        feature_mask_history.append(feature_mask.clone())

        # Let AFA method select features until it chooses to stop
        # or until all features are observed
        while True:
            # Call the AFA method
            selection = afa_method(features, feature_mask)

            # If the AFA method chooses to stop, break
            if selection == 0:
                break

            # Otherwise, update the feature mask
            feature_mask[selection] = True
            feature_mask_history.append(feature_mask.clone())

            # If all features have been selected, stop
            if feature_mask.all():
                break

        # Now we evaluate how good the selected features are
        # The metric is always a function of which features we selected, and what the true label is
        metrics: dict = evaluator(feature_mask_history, label)

        # Update the metrics
        all_metrics.append(metrics)

    # Average each metric over all samples
    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = sum([metrics[key] for metrics in all_metrics]) / len(
            all_metrics
        )

    # Print the average metrics
    print("Average metrics:")
    for key, value in avg_metrics.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
