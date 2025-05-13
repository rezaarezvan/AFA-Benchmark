"""
Evaluates all static feature selection methods (CAE or permutation),
saving metrics in the same format as the afa evaluation script.
"""
import argparse
from pathlib import Path
import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from typing import Any, Optional
from tqdm import tqdm
import yaml
import numpy as np
from coolname import generate_slug
from torchmetrics import AUROC
from sklearn.metrics import accuracy_score, f1_score
from static.models import get_network, BaseModel
from static.static_methods import DifferentiableSelector, ConcreteMask
from common.utils import set_seed
from common.custom_types import AFADataset
from common.registry import AFA_DATASET_REGISTRY, STATIC_METHOD_REGISTRY
from common.custom_types import AFADataset


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

def make_loader(ds, sel, shuffle=True, batch_size=128, drop_last=True):
    ds_sel = TensorDataset(ds.features[:, sel], ds.labels)
    return DataLoader(ds_sel, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)

def evaluator(
    feature_mask_history_all,
    prediction_history_all,
    labels_all,
) -> dict[str, Any]:
    labels = torch.stack(labels_all)
    labels = torch.argmax(labels, dim=1).cpu().numpy()

    accuracy_all, f1_all = aggregate_metrics(prediction_history_all, labels)

    return {
        "accuracy_all": accuracy_all,
        "f1_all": f1_all,
        "feature_mask_history_all": [
            [mask.cpu() for mask in sample_masks] for sample_masks in feature_mask_history_all
        ],
    }


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


def main(args):
    eval_seed = 42
    set_seed(eval_seed)
    device = torch.device(args.device)

    for method in STATIC_METHOD_REGISTRY:
        print(f"Start evaluating {method} method on {args.dataset_type} dataset")
        for split in list(range(1, 6)):
            print(f"Dataset split {split}")
            train_dataset_path = Path(f"data/{args.dataset_type}/train_split_{split}.pt")
            val_dataset_path = Path(f"data/{args.dataset_type}/val_split_{split}.pt")
            test_dataset_path = Path(f"data/{args.dataset_type}/{args.dataset_fraction_name}_split_{split}.pt")
            train_dataset: AFADataset = AFA_DATASET_REGISTRY[args.dataset_type].load(train_dataset_path)
            val_dataset:   AFADataset = AFA_DATASET_REGISTRY[args.dataset_type].load(val_dataset_path)
            test_dataset:  AFADataset = AFA_DATASET_REGISTRY[args.dataset_type].load(test_dataset_path)

            labels_all = []
            for _, label in iter(test_dataset):
                labels_all.append(label)
            
            for ds in (train_dataset, val_dataset, test_dataset):
                ds.features = ds.features.float()
                ds.labels   = ds.labels.argmax(dim=1).long()

            d_in  = train_dataset.features.shape[-1]
            d_out = int(train_dataset.labels.max().item()) + 1

            # TODO should we use d_in? also can only start from 1, not 0
            budgets = list(range(1, d_in + 1))

            n_samples = len(test_dataset)
            B = len(budgets)
            feature_mask_history_all: list[list[Optional[Tensor]]] = [[None]*B for _ in range(n_samples)]
            prediction_history_all: list[list[Optional[Tensor]]] = [[None]*B for _ in range(n_samples)]

            auroc_metric = lambda pred, y: AUROC(task="multiclass", num_classes=d_out)(
                pred.softmax(dim=1), y
            )

            # select features & retrain for each budget
            for j, budget in enumerate(tqdm(budgets, desc="budgets")):
                if method == "cae":
                    model = get_network(d_in, d_out).to(device)
                    selector_layer = ConcreteMask(d_in, budget)
                    diff = DifferentiableSelector(model, selector_layer).to(device)
                    diff.fit(
                        train_loader=DataLoader(
                            TensorDataset(train_dataset.features, train_dataset.labels),
                            batch_size=128,
                            shuffle=True,
                            drop_last=True,
                        ),
                        val_loader=DataLoader(
                            TensorDataset(val_dataset.features, val_dataset.labels),
                            batch_size=1024,
                        ),
                        lr=1e-3,
                        nepochs=250,
                        loss_fn=torch.nn.CrossEntropyLoss(),
                        patience=5,
                        verbose=False,
                    )
                    # Extract top features.
                    logits = selector_layer.logits.cpu().data.numpy()
                    selected_features = np.sort(logits.argmax(axis=1))
                    
                    unique = np.unique(selected_features)
                    if len(unique) != budget:
                        num_extras = budget - len(unique)
                        remaining_features = np.setdiff1d(np.arange(d_in), unique)
                        selected_features = np.sort(np.concatenate([unique, remaining_features[:num_extras]]))
                    selected = selected_features
                elif method == "permutation":
                    base = BaseModel(get_network(d_in, d_out).to(device))
                    base.fit(
                        train_loader=DataLoader(
                            TensorDataset(train_dataset.features, train_dataset.labels),
                            batch_size=128,
                            shuffle=True,
                            drop_last=True,
                        ),
                        val_loader=DataLoader(
                            TensorDataset(val_dataset.features, val_dataset.labels),
                            batch_size=1024,
                        ),
                        lr=1e-3,
                        nepochs=250,
                        loss_fn=torch.nn.CrossEntropyLoss(),
                        verbose=False,
                    )
                    # compute importances
                    permutation_importance = np.zeros(d_in)
                    X_train = train_dataset.features
                    for feat in range(d_in):
                        X_val = val_dataset.features.clone()
                        # permute feature feat
                        X_val[:, feat] = X_train[torch.randint(len(X_train), (len(X_val),)), feat]
                        with torch.no_grad():
                            p = base.model(X_val.to(device)).cpu()
                        permutation_importance[feat] = -auroc_metric(p, val_dataset.labels)
                    ranked = np.argsort(permutation_importance)[::-1].copy()
                    selected = ranked[:budget]
                else:
                    raise NotImplementedError("Static feature selection method not implemented")

                # retrain classifier on selected features
                train_subset_loader = make_loader(train_dataset, selected)
                val_subset_loader   = make_loader(val_dataset, selected)

                best_model = None
                best_loss = float("inf")
                for _ in range(args.num_restarts):
                    model = BaseModel(get_network(budget, d_out).to(device))
                    model.fit(train_subset_loader, val_subset_loader, lr=1e-3, nepochs=250, loss_fn=torch.nn.CrossEntropyLoss(), verbose=False)
                    val_loss = model.evaluate(val_subset_loader, torch.nn.CrossEntropyLoss())
                    if val_loss < best_loss:
                        best_loss = val_loss
                        best_model = model
                assert best_model is not None

                test_loader = make_loader(test_dataset, selected, shuffle=False, batch_size=1, drop_last=False)
                for i, (x_sel, _) in enumerate(test_loader):
                    with torch.no_grad():
                        logits = best_model.model(x_sel.to(device)).squeeze(0).cpu()
                    feature_mask_history_all[i][j] = torch.zeros(d_in, dtype=torch.bool, device="cpu").scatter_(0, torch.tensor(selected), True)
                    prediction_history_all[i][j] = logits

            metrics = evaluator(feature_mask_history_all, prediction_history_all, labels_all)

            write_eval_results(
                args.results_folder / f"static_{method}_{args.dataset_type}_{split}",
                metrics,
                method,
                __file__,
                {
                    "dataset_type": args.dataset_type,
                    "hard_budget": d_in,
                    "pretrained_model_path": None,
                    "seed": 42,
                    "train_dataset_path": str(train_dataset_path),
                    "val_dataset_path": str(val_dataset_path),
                },
                True,
                None,
                None,
                None,
                {
                    "seed": eval_seed,
                    "dataset_path": str(test_dataset_path),
                    "hard_budget": d_in,
                },
                args.dataset_type,
            )


if __name__ == "__main__":
    # TODO maybe should use configuration file to control training?
    parser = argparse.ArgumentParser(description="Evaluate static feature selection methods")
    parser.add_argument("--dataset_type", type=str, required=True, choices=AFA_DATASET_REGISTRY.keys())
    parser.add_argument("--dataset_fraction_name", type=str, default="test")
    parser.add_argument("--num_restarts", type=int, default=5)
    parser.add_argument("--results_folder", type=Path, default=Path("results"))
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    main(args)
