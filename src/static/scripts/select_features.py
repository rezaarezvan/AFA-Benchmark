import argparse
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm
from torchmetrics import AUROC
from torch.utils.data import DataLoader, TensorDataset
from static.models import get_network, BaseModel
from static.static_methods import DifferentiableSelector, ConcreteMask
from common.utils import set_seed
from common.registry import AFA_DATASET_REGISTRY, STATIC_METHOD_REGISTRY
from common.custom_types import AFADataset


def compute_selected_for_budget(method, d_in, d_out, train_ds, val_ds, budget, device):
    if method == "cae":
        selector = ConcreteMask(d_in, budget)
        diff = DifferentiableSelector(get_network(d_in, d_out), selector).to(device)
        diff.fit(
            train_loader=DataLoader(
                TensorDataset(train_ds.features, train_ds.labels),
                batch_size=128,
                shuffle=True,
                drop_last=True,
            ),
            val_loader=DataLoader(
                TensorDataset(val_ds.features, val_ds.labels), batch_size=1024
            ),
            lr=1e-3,
            nepochs=250,
            loss_fn=torch.nn.CrossEntropyLoss(),
            patience=5,
            verbose=False,
        )
        logits = selector.logits.cpu().data.numpy()
        selected = np.sort(logits.argmax(1))
        unique = np.unique(selected)
        if len(unique) != budget:
            num_extras = budget - len(unique)
            remaining_features = np.setdiff1d(np.arange(d_in), unique)
            selected = np.sort(
                np.concatenate([unique, remaining_features[:num_extras]])
            )
    elif method == "permutation":
        base = BaseModel(get_network(d_in, train_ds.labels.max().item() + 1).to(device))
        base.fit(
            train_loader=DataLoader(
                TensorDataset(train_ds.features, train_ds.labels),
                batch_size=128,
                shuffle=True,
                drop_last=True,
            ),
            val_loader=DataLoader(
                TensorDataset(val_ds.features, val_ds.labels), batch_size=1024
            ),
            lr=1e-3,
            nepochs=250,
            loss_fn=torch.nn.CrossEntropyLoss(),
            verbose=False,
        )
        # compute importances
        auroc = lambda p, y: AUROC(task="multiclass", num_classes=y.max().item() + 1)(
            p.softmax(1), y
        )
        permutation_importance = np.zeros(d_in)
        for f in range(d_in):
            X_val = val_ds.features.clone()
            X_val[:, f] = train_ds.features[
                torch.randint(len(train_ds), (len(X_val),)), f
            ]
            with torch.no_grad():
                p = base.model(X_val.to(device)).cpu()
            permutation_importance[f] = -auroc(p, val_ds.labels)
        ranked = np.argsort(permutation_importance)[::-1]
        selected = ranked[:budget]
    else:
        raise ValueError(f"Unknown method {method}")
    return selected.tolist()


def main(args):
    set_seed(42)
    device = torch.device(args.device)

    for method in STATIC_METHOD_REGISTRY:
        # load datasets
        paths = {
            "train": Path(f"data/{args.dataset_type}/train_split_{args.split}.pt"),
            "val": Path(f"data/{args.dataset_type}/val_split_{args.split}.pt"),
        }
        train_ds: AFADataset = AFA_DATASET_REGISTRY[args.dataset_type].load(
            paths["train"]
        )
        val_ds: AFADataset = AFA_DATASET_REGISTRY[args.dataset_type].load(paths["val"])
        # preprocess
        for ds in (train_ds, val_ds):
            ds.features = ds.features.float()
            ds.labels = ds.labels.argmax(1).long()

        d_in = train_ds.features.shape[-1]
        d_out = train_ds.labels.shape[-1]
        s = args.start_feature_num
        e = args.end_feature_num or d_in
        if s < 1 or e < s:
            raise ValueError(
                f"Invalid budget range: start_feature_num={s}, end_feature_num={e}"
            )
        e = min(e, d_in)
        budgets = list(range(s, e + 1))

        selected_history = []
        for b in tqdm(budgets, desc=f"{method}-{args.dataset_type}-split{args.split}"):
            sel = compute_selected_for_budget(
                method, d_in, d_out, train_ds, val_ds, b, device
            )
            selected_history.append(sel)

        # save
        out = (
            args.results_folder
            / f"{method}_{args.dataset_type}_split{args.split}_s{s}_e{e}"
        )
        out.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"budgets": budgets, "selected_history": selected_history},
            out / "selected.pt",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_type", required=True, choices=AFA_DATASET_REGISTRY.keys()
    )
    parser.add_argument("--start_feature_num", type=int, default=1)
    parser.add_argument("--end_feature_num", type=int, required=True)
    parser.add_argument(
        "--results_folder", type=Path, default=Path("models/static/selected")
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--split", type=int, default=1)
    args = parser.parse_args()
    main(args)
