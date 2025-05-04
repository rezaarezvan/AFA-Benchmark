import argparse
import yaml
import time
from pathlib import Path
from copy import deepcopy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from coolname import generate_slug
from rich import print as rprint
from common.utils import set_seed
from common.custom_types import AFADataset
from common.classifiers import NNClassifier
from common.registry import AFA_DATASET_REGISTRY

def generate_uniform_mask(batch_size, feature_dim, p_miss=0.2):
    return (torch.rand(batch_size, feature_dim) > p_miss).float()

def train_model(classifier: NNClassifier, train_loader, val_loader, device, num_epochs=100, lr=1e-3, patience=5):
    model = classifier.predictor
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=patience, min_lr=1e-6)
    acc_metric = Accuracy(task='multiclass', num_classes=classifier.output_dim)

    best_model_state = None
    bad_epochs = 0

    for epoch in range(num_epochs):
        model.train()

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            m = generate_uniform_mask(len(x), x.shape[1]).to(device)
            x_masked = x * m
            logits = classifier(x_masked, m)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        preds, targets = [], []

        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                m_val = generate_uniform_mask(len(x_val), x_val.shape[1]).to(device)
                logits_val = classifier(x_val, m_val)
                preds.append(logits_val.argmax(dim=1).cpu())
                targets.append(y_val.cpu())

        acc = acc_metric(torch.cat(preds), torch.cat(targets))
        print(f"[Epoch {epoch+1}] val_acc={acc:.4f}")

        scheduler.step(acc)

        if acc == scheduler.best:
            best_model_state = deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= patience:
            print("Early stopping.")
            break

    if best_model_state:
        model.load_state_dict(best_model_state)

def main(dataset_type: str, split: int, seed: int, config: dict, classifier_folder: Path, device: torch.device):
    # Load datasets
    train_dataset_path=Path(f"data/{dataset_type}/train_split_{split}.pt")
    val_dataset_path=Path(f"data/{dataset_type}/val_split_{split}.pt")

    train_dataset: AFADataset = AFA_DATASET_REGISTRY[dataset_type].load(train_dataset_path)
    val_dataset: AFADataset = AFA_DATASET_REGISTRY[dataset_type].load(val_dataset_path)

    d_in = train_dataset.features.shape[-1]
    d_out = train_dataset.labels.shape[-1]

    for ds in (train_dataset, val_dataset):
        ds.features = ds.features.float()
        ds.labels = ds.labels.argmax(dim=1).long()

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], pin_memory=True)

    classifier = NNClassifier(input_dim=d_in, output_dim=d_out, device=device)

    # Train
    train_model(
        classifier,
        train_loader,
        val_loader,
        device,
        num_epochs=config["epochs"],
        lr=config["lr"],
        patience=config["patience"]
    )

    # Save
    slug = generate_slug(2)
    save_path = classifier_folder / dataset_type / f"split_{split}" / f"seed_{seed}" / f"{slug}.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    classifier.save(save_path)
    print(f"Classifier saved to: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pipeline_config_path", 
        type=Path,
        default=Path("configs/classifiers/pipeline.yml"),
        help="Path to pipeline.yml"
    )
    parser.add_argument(
        "--train_config_path", 
        type=Path,
        default=Path("configs/classifiers/train_nnclassifier.yml"),
        help="Path to classifier training config"
    )
    parser.add_argument(
        "--classifier_folder", 
        type=Path, 
        default=Path(f"models/classifiers/{time.strftime('%Y-%m-%d_%H-%M-%S')}"), 
        help="Folder to save trained classifiers"
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    args = parser.parse_args()

    # Validate paths
    assert args.pipeline_config_path.exists(), "Slurm pipeline config path does not exist"
    assert args.train_config_path.exists(), "Train config path does not exist"

    # Load configs
    with open(args.pipeline_config_path, "r") as f:
        pipeline_cfg = yaml.safe_load(f)

    with open(args.train_config_path, "r") as f:
        train_cfg = yaml.safe_load(f)

    dataset_types = pipeline_cfg["dataset_types"]
    splits = pipeline_cfg["dataset_splits"]
    seeds = pipeline_cfg["seeds"]

    for dataset_type in dataset_types:
        for split in splits:
            for seed in seeds:
                set_seed(seed)
                rprint(f"[bold green]Training classifier for {dataset_type} | split {split} | seed {seed}[/bold green]")
                torch.cuda.empty_cache()
                main(dataset_type, split, seed, train_cfg, args.classifier_folder, torch.device(args.device))
                torch.cuda.empty_cache()