"""
Evaluates all trained AFA methods.
"""

import argparse
from pathlib import Path
from textwrap import indent
from typing import Any

from numpy.typing import NDArray
import torch
from torch import Tensor
from tqdm import tqdm
import yaml

from common.custom_types import (
    AFAClassifier,
    AFAClassifierFn,
    AFADataset,
    AFAMethod,
    FeatureMask,
    Label,
)
from common.registry import (
    AFA_CLASSIFIER_REGISTRY,
    AFA_DATASET_REGISTRY,
    AFA_METHOD_REGISTRY,
)
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

from common.utils import set_seed
from eval.utils import get_classifier_paths_trained_on_data

from coolname import generate_slug


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

    # Save metrics as NDArray
    return {
        "accuracy_all": accuracy_all.detach().cpu(),
        "f1_all": f1_all.detach().cpu(),
        "feature_mask_history_all": [
            [t.detach().cpu() for t in sublist] for sublist in feature_mask_history_all
        ],
    }


def eval_afa_method(
    afa_method: AFAMethod,
    dataset: AFADataset,
    hard_budget: int,
    afa_classifier_fn: AFAClassifierFn,
) -> dict[str, Any]:
    """Evaluate an AFA method.

    Args:
        afa_method (AFAMethod): The AFA method to evaluate.
        dataset (AFADataset): The dataset to evaluate on.
        hard_budget (int): The number of features to select.
        classifier_fn (AFAClassifierFn): The classifier function to use for evaluation.

    Returns:
        dict[str, float]: A dictionary containing the evaluation results.

    """
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

        # We will keep a history of which features have been observed, in case its relevant for evaluation
        feature_mask_history: list[FeatureMask] = []

        # And also a history of predictions
        prediction_history: list[Label] = []

        # Start with all features unobserved
        feature_mask = torch.zeros_like(features, dtype=torch.bool)
        masked_features = features.clone()
        masked_features[~feature_mask] = 0.0

        # Let AFA method select features for a fixed number of steps
        for _ in range(hard_budget):
            # Always calculate a prediction
            prediction = afa_classifier_fn(
                masked_features.unsqueeze(0), feature_mask.unsqueeze(0)
            ).squeeze(0)

            prediction_history.append(prediction)

            # Select new features
            selection = afa_method.select(
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

    # Now we have a history of feature masks and predictions for each sample in the dataset
    eval_results = evaluator(
        feature_mask_history_all, prediction_history_all, labels_all
    )

    return eval_results


def write_eval_results(
    folder_path: Path,
    metrics: dict[str, Any],
    method_type: str,
    method_path: str,
    method_params: dict[str, Any],
    is_builtin_classifier: bool,
    classifier_type: str | None,
    classifier_path: str | None,
    classifier_params: dict[str, Any] | None,
    eval_params: dict[str, Any],
    dataset_type: str,
):
    """Write the evaluation results to a folder with two files: results.pt and params.yml."""
    folder_path.mkdir(parents=True, exist_ok=True)

    # Save the metrics to a .pt file
    torch.save(
        metrics,
        folder_path / "results.pt",
    )

    # Write to params.yml
    with open(
        folder_path / "params.yml",
        "w",
    ) as file:
        # Don't write the dataset type for method/classifier since we assume that the same dataset type is used.
        yaml.dump(
            {
                **{
                    f"method_{k}": v
                    for k, v in method_params.items()
                    if k != "dataset_type"
                },
                "method_type": method_type,
                "method_path": method_path,
                **(
                    {
                        f"classifier_{k}": v
                        for k, v in classifier_params.items()
                        if k != "dataset_type"
                    }
                    if classifier_params
                    else {}
                ),
                **({"classifier_type": classifier_type} if classifier_type else {}),
                **(
                    {"classifier_path": str(classifier_path)} if classifier_path else {}
                ),
                "is_builtin_classifier": is_builtin_classifier,
                **{f"eval_{k}": v for k, v in eval_params.items()},
                "dataset_type": dataset_type,
            },
            file,
            default_flow_style=False,
        )


def main(
    method_folder: Path,
    classifier_folder: Path,
    results_folder: Path,
    dataset_fraction_name: str,
):
    # Currently, always use a constant seed during evaluation
    eval_seed = 42
    set_seed(eval_seed)

    # Loop through each method type in the models folder
    for method_type_path in method_folder.iterdir():
        method_type = method_type_path.name
        method_cls = AFA_METHOD_REGISTRY[method_type]
        # Loop through each trained method
        for trained_method_path in method_type_path.iterdir():
            # trained_instance_name = trained_method_path.name
            # There should be two files in each directory: model.pt and params.yml

            # model.pt can be used to load the AFA method
            saved_model_path = trained_method_path / "model.pt"
            afa_method = method_cls.load(saved_model_path, torch.device("cpu"))

            # The params.yml file should contain the hard budget and dataset paths
            method_params_method = trained_method_path / "params.yml"

            # Params file should contain the hard budget and dataset paths
            # Open it as yaml
            with open(method_params_method, "r") as file:
                method_params_dict: dict = yaml.safe_load(file)

            # Use the same hard budget during evaluation as during training
            # Note that this can be None, in which case we will use the maximum number of features in the dataset
            # during evaluation
            hard_budget = method_params_dict["hard_budget"]

            # The dataset we want to use during evaluation should be the same split as the one used during training,
            # but possibly using a different fraction of the dataset (i.e. val or test)
            train_dataset_path = Path(method_params_dict["train_dataset_path"])
            eval_dataset_type = train_dataset_path.parent.name
            eval_dataset_name = train_dataset_path.name.replace(
                "train", dataset_fraction_name
            )
            eval_dataset_path = train_dataset_path.parent / eval_dataset_name
            eval_dataset = AFA_DATASET_REGISTRY[eval_dataset_type].load(
                eval_dataset_path
            )
            assert len(eval_dataset.features) > 0
            assert len(eval_dataset.labels) > 0
            print(
                f"Loaded AFA method {method_type} from {trained_method_path} trained on {train_dataset_path}"
            )
            print(f"  Using dataset {eval_dataset_path} for evaluation")

            # Loop through each classifier type in the classifier folder
            classifier_folder.mkdir(parents=True, exist_ok=True)
            for i, classifier_type_path in enumerate(classifier_folder.iterdir()):
                classifier_type = classifier_type_path.name
                # Get paths to all classifiers of this type that were trained on the same dataset
                classifier_paths = get_classifier_paths_trained_on_data(
                    classifier_type, train_dataset_path, classifier_folder
                )
                # There should only be one classifier (for each classifier type) that was trained on this dataset
                assert len(classifier_paths) == 1, (
                    f"Found {len(classifier_paths)} classifiers of type {classifier_type} trained on {train_dataset_path}, expected 1"
                )
                classifier_path = classifier_paths[0]
                print(
                    f"  Using classifier {classifier_type} at {classifier_path} trained on {train_dataset_path}..."
                )

                # Load classifier
                classifier = AFA_CLASSIFIER_REGISTRY[classifier_type].load(
                    classifier_path / "model.pt", torch.device("cpu")
                )

                # Load classifier params
                with open(classifier_path / "params.yml", "r") as file:
                    classifier_params_dict: dict = yaml.safe_load(file)

                # Do the evaluation
                metrics = eval_afa_method(
                    afa_method,
                    eval_dataset,
                    hard_budget if hard_budget else eval_dataset.features.shape[-1],
                    classifier,
                )

                # Write all results
                write_eval_results(
                    results_folder / generate_slug(2),
                    metrics,
                    method_type,
                    str(trained_method_path),
                    method_params_dict,
                    False,
                    classifier_type,
                    str(classifier_path),
                    classifier_params_dict,
                    {
                        "seed": eval_seed,
                        "dataset_path": str(eval_dataset_path),
                        "hard_budget": hard_budget,
                    },
                    eval_dataset_type,
                )

            # Finally, use the built-in classifier of the AFA method
            print(f"  Using built-in classifier...")
            metrics = eval_afa_method(
                afa_method, eval_dataset, hard_budget, afa_method.predict
            )

            write_eval_results(
                results_folder / generate_slug(2),
                metrics,
                method_type,
                str(trained_method_path),
                method_params_dict,
                True,
                None,
                None,
                None,
                {
                    "seed": eval_seed,
                    "dataset_path": str(eval_dataset_path),
                    "hard_budget": hard_budget,
                },
                eval_dataset_type,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate all trained AFA methods",
    )
    parser.add_argument(
        "--method_folder",
        type=Path,
        default="models/methods",
        help="Path to the method folder",
    )
    parser.add_argument(
        "--classifier_folder",
        type=Path,
        default="models/classifiers",
        help="Path to the classifier folder",
    )
    parser.add_argument(
        "--results_folder",
        type=Path,
        default="results",
        help="Path to the evaluation results folder",
    )
    parser.add_argument(
        "--dataset_fraction_name",
        type=str,
        default="val",
        help="Which part of the dataset to use for evaluation. Commonly 'val' or 'test'",
    )
    args = parser.parse_args()

    main(
        args.method_folder,
        args.classifier_folder,
        args.results_folder,
        args.dataset_fraction_name,
    )
