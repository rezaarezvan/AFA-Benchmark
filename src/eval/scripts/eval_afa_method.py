"""
Evaluates an AFA method (as defined in common.custom_types) on a specific dataset.
"""

import argparse
import os

import torch
from tqdm import tqdm

from common.custom_types import (
    AFADataset,
    AFAMethod,
    FeatureMask,
    Label,
)
from common.registry import AFA_DATASET_REGISTRY, AFA_METHOD_REGISTRY


def evaluator(
    feature_mask_history_all: list[list[FeatureMask]],
    prediction_history_all: list[list[Label]],
    labels_all: list[Label],
) -> dict:
    """
    Calculates accuracy (using last prediction before afa was stopped) and average number of features selected for each sample in the dataset.
    """
    assert (
        len(feature_mask_history_all) == len(prediction_history_all) == len(labels_all)
    ), "All three lists must have the same length"

    num_samples = len(feature_mask_history_all)

    # Calculate accuracy
    correct_predictions = 0
    for prediction_history, label in zip(prediction_history_all, labels_all):
        # Get the last prediction for this sample
        prediction = prediction_history[-1]
        if prediction.argmax(-1) == label.argmax(-1):
            correct_predictions += 1

    accuracy = correct_predictions / num_samples

    # Calculate average number of features selected
    total_features_selected = 0
    for feature_mask_history in feature_mask_history_all:
        total_features_selected += len(feature_mask_history)

    avg_features_selected = total_features_selected / num_samples

    return {
        "accuracy": accuracy,
        "avg_features_selected": avg_features_selected,
    }


def eval_afa_method(args: argparse.Namespace) -> dict[str, float]:
    """
    Evaluates an AFA method on a specific dataset and returns a dictionary of metrics.
    """

    # Load the AFA method
    afa_method: AFAMethod = AFA_METHOD_REGISTRY[args.afa_method_name].load(
        args.afa_method_path
    )
    print(f"Loaded AFA method {args.afa_method_name} from {args.afa_method_path}")

    # Load the dataset
    dataset: AFADataset = AFA_DATASET_REGISTRY[args.dataset_type].load(
        args.dataset_val_path
    )
    print(f"Loaded dataset {args.dataset_type} from {args.dataset_val_path}")

    # Store feature mask history, label prediction history, and true label for each sample in the dataset
    feature_mask_history_all: list[list[FeatureMask]] = []
    prediction_history_all: list[list[Label]] = []
    labels_all: list[Label] = []

    # Loop over the dataset
    for data in tqdm(iter(dataset), total=len(dataset), desc="Evaluating"):
        # Each datasample has a vector of features and a class (label)
        features, label = data

        # Immediately store the true label for this sample
        labels_all.append(label)

        # AFA methods expect a batch dimension
        # features: Features = features.unsqueeze(0)
        # label: Label = label.unsqueeze(0)

        # We will keep a history of which features have been observed, in case its relevant for evaluation
        feature_mask_history: list[FeatureMask] = []

        # And also a history of predictions
        prediction_history: list[Label] = []

        # Start with all features unobserved
        feature_mask = torch.zeros_like(features, dtype=torch.bool)

        # Let AFA method select features until it chooses to stop
        # or until all features are observed
        while True:
            # Always calculate a prediction
            prediction = afa_method.predict(
                features.unsqueeze(0), feature_mask.unsqueeze(0)
            ).squeeze(0)

            prediction_history.append(prediction)

            # Select new features or stop
            selection = afa_method.select(
                features.unsqueeze(0), feature_mask.unsqueeze(0)
            ).squeeze(0)

            # If the AFA method chooses to stop, break
            if selection == 0:
                break

            # Otherwise, update the feature mask
            feature_mask[selection - 1] = True
            feature_mask_history.append(feature_mask.clone())

            # If all features have been selected, stop
            if feature_mask.all():
                break

        # Add the feature mask history and prediction history of this sample to the overall history
        feature_mask_history_all.append(feature_mask_history)
        prediction_history_all.append(prediction_history)

    # Now we have a history of feature masks and predictions for each sample in the dataset
    eval_results = evaluator(
        feature_mask_history_all, prediction_history_all, labels_all
    )

    return eval_results


if __name__ == "__main__":
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
        "--dataset_type",
        type=str,
        required=True,
        help="Name of the AFADataset to evaluate. Must be one of: "
        + ", ".join(AFA_DATASET_REGISTRY.keys()),
    )
    parser.add_argument(
        "--dataset_val_path",
        type=str,
        required=True,
        help="Path to a .pt file containing the validation AFADataset to evaluate the AFAMethod on",
    )
    parser.add_argument(
        "--eval_save_path",
        type=str,
        required=True,
        help="Path to a .pt file to save the evaluation results",
    )
    args = parser.parse_args()

    if args.afa_method_name not in AFA_METHOD_REGISTRY:
        raise ValueError(
            f"Method {args.afa_method_name} not in registry. Must be one of: "
            + ", ".join(AFA_METHOD_REGISTRY.keys())
        )
    if args.dataset_type not in AFA_DATASET_REGISTRY:
        raise ValueError(
            f"Method {args.afa_dataset_type} not in registry. Must be one of: "
            + ", ".join(AFA_DATASET_REGISTRY.keys())
        )

    eval_results = eval_afa_method(args)

    # Save the metrics to a file
    os.makedirs(os.path.dirname(args.eval_save_path), exist_ok=True)
    torch.save(eval_results, args.eval_save_path)
    print(f"Saved metrics to {args.eval_save_path}")
