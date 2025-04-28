"""
Evaluates all trained AFA methods.
"""

import argparse
from pathlib import Path

import torch
from tqdm import tqdm
import yaml

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
    Calculates
      • accuracy using the **final** prediction of every sample
      • per-step accuracy (``accuracy_all``) so it lines up with the
        longest prediction history in the batch
      • the average number of feature masks produced per sample
    """
    assert (
        len(feature_mask_history_all) == len(prediction_history_all) == len(labels_all)
    ), "All three lists must have the same length"

    num_samples = len(prediction_history_all)

    # ------------------------------------------------------------------
    # 1) final-step accuracy (identical to the original implementation)
    # ------------------------------------------------------------------
    correct_final = sum(
        1
        for preds, lbl in zip(prediction_history_all, labels_all)
        if preds[-1].argmax(-1) == lbl.argmax(-1)
    )
    accuracy = correct_final / num_samples

    # ------------------------------------------------------------------
    # 2) per-step accuracy across **all** available predictions
    # ------------------------------------------------------------------
    max_len = max(len(preds) for preds in prediction_history_all)
    accuracy_all: list[float] = []

    for step_idx in range(max_len):
        correct, total = 0, 0
        for preds, lbl in zip(prediction_history_all, labels_all):
            if step_idx < len(preds):           # sample has a prediction at this step
                total += 1
                if preds[step_idx].argmax(-1) == lbl.argmax(-1):
                    correct += 1
        accuracy_all.append(correct / total if total else 0.0)

    # ------------------------------------------------------------------
    # 3) number of features selected per sample
    # ------------------------------------------------------------------

    num_features_selected = [len(mask_hist) for mask_hist in feature_mask_history_all]

    return {
        "accuracy": accuracy,
        "accuracy_all": accuracy_all,
        "num_features_selected": num_features_selected,
        "feature_mask_history_all": feature_mask_history_all,
    }


def eval_afa_method(afa_method: AFAMethod, dataset: AFADataset, hard_budget: int) -> dict[str, float]:
    """
    Evaluates an AFA method on a specific dataset and hard budget, and returns a dictionary of metrics.
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
            prediction = afa_method.predict(
                masked_features.unsqueeze(0), feature_mask.unsqueeze(0)
            ).squeeze(0)

            prediction_history.append(prediction)

            # Select new features
            selection = afa_method.select(
                masked_features.unsqueeze(0), feature_mask.unsqueeze(0)
            ).squeeze(0)

            # Update the feature mask
            feature_mask[selection - 1] = True
            masked_features[~feature_mask] = 0.0
            feature_mask_history.append(feature_mask.clone())

        # Add the feature mask history and prediction history of this sample to the overall history
        feature_mask_history_all.append(feature_mask_history)
        prediction_history_all.append(prediction_history)

    # Now we have a history of feature masks and predictions for each sample in the dataset
    eval_results = evaluator(
        feature_mask_history_all, prediction_history_all, labels_all
    )

    return eval_results


def main(model_folder: Path, results_folder: Path, dataset_fraction_name: str):
    # Loop through each method type in the models folder
    for method_type_path in model_folder.iterdir():
        method_name = method_type_path.name
        method_cls = AFA_METHOD_REGISTRY[method_name]
        # Loop through each trained instance
        for trained_instance_path in method_type_path.iterdir():
            trained_instance_name = trained_instance_path.name
            # There should be two files in each directory: model.pt and params.yml

            # model.pt can be used to load the AFA method
            saved_model_path = trained_instance_path / "model.pt"
            afa_method = method_cls.load(saved_model_path, torch.device("cpu"))
            print(f"Loaded AFA method {method_name} from {trained_instance_path}")

            # The params.yml file should contain the hard budget and dataset paths
            params_path = trained_instance_path / "params.yml"

            # Params file should contain the hard budget and dataset paths
            # Open it as yaml
            with open(params_path, "r") as file:
                params_dict: dict = yaml.safe_load(file)
            # Use the same hard budget during evaluation as during training
            hard_budget = params_dict["hard_budget"]

            # The dataset we want to use during evaluation should be the same split as the one used during training,
            # but possible using a different fraction of the dataset (i.e. val or test)
            train_dataset_path = Path(params_dict["train_dataset_path"])
            dataset_type = train_dataset_path.parent.name
            eval_dataset_name = train_dataset_path.name.replace("train", dataset_fraction_name)
            eval_dataset_path = train_dataset_path.parent / eval_dataset_name
            eval_dataset = AFA_DATASET_REGISTRY[dataset_type].load(
                eval_dataset_path
            )
            print(f"Loaded dataset {dataset_type} from {eval_dataset_path}")

            # Do the evaluation
            metrics = eval_afa_method(afa_method, eval_dataset, hard_budget)

            # Save the metrics to the results folder
            (results_folder / method_name / trained_instance_name).mkdir(
                parents=True, exist_ok=True
            )

            # Save the metrics to a .pt file
            torch.save(
                metrics,
                results_folder / method_name / trained_instance_name / "results.pt",
            )

            # The only config parameter in evaluation is which dataset we used and the hard budget (though it should be the same as during training)
            with open(
                results_folder / method_name / trained_instance_name / "params.yml",
                "w",
            ) as file:
                yaml.dump(
                    {
                        "eval_dataset_path": str(eval_dataset_path),
                        "hard_budget": hard_budget,
                    },
                    file,
                    default_flow_style=False,
                )



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate all trained AFA methods",
    )
    parser.add_argument(
        "--models_folder",
        type=Path,
        default="models",
        help="Path to the models folder",
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

    main(args.models_folder, args.results_folder, args.dataset_fraction_name)
