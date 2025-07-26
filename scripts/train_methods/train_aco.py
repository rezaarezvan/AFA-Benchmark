import torch
import hydra
import wandb
import logging
import numpy as np
import torch.nn.functional as F

from pathlib import Path
from typing import Any, cast
from omegaconf import OmegaConf
from eval.utils import plot_metrics
from tempfile import TemporaryDirectory
from eval.metrics import eval_afa_method
from afa_oracle import create_aaco_method
from common.config_classes import AACOTrainConfig
from common.utils import load_dataset_artifact, set_seed

log = logging.getLogger(__name__)


def load_original_mnist_data(mnist_pkl_path="MNIST.pkl"):
    """
    Load the original authors' MNIST.pkl data with their exact preprocessing
    """
    log.info(f"Loading original MNIST data from {mnist_pkl_path}")

    # Load the pickle file
    mnist = np.load(mnist_pkl_path, allow_pickle=True)

    # Extract train and validation data
    mnist_train = mnist.get('train')
    mnist_val = mnist.get('valid')

    # Handle both numpy and tensor formats
    if isinstance(mnist_train[0], np.ndarray):
        X_train = torch.from_numpy(mnist_train[0]).float()
    else:
        X_train = mnist_train[0].float()

    if isinstance(mnist_train[1], np.ndarray):
        y_train = torch.from_numpy(mnist_train[1])
    else:
        y_train = mnist_train[1]

    if isinstance(mnist_val[0], np.ndarray):
        X_valid = torch.from_numpy(mnist_val[0]).float()
    else:
        X_valid = mnist_val[0].float()

    if isinstance(mnist_val[1], np.ndarray):
        y_valid = torch.from_numpy(mnist_val[1])
    else:
        y_valid = mnist_val[1]

    log.info(
        f"Raw data shapes - X_train: {X_train.shape}, y_train: {y_train.shape}")
    log.info(
        f"Raw data ranges - X_train: [{X_train.min():.3f}, {X_train.max():.3f}]")

    # Check if labels are already one-hot encoded
    if len(y_train.shape) > 1 and y_train.shape[1] > 1:
        log.info("Labels are already one-hot encoded")
        y_train = y_train.float()
        y_valid = y_valid.float()
        n_classes = y_train.shape[1]
    else:
        log.info("Converting labels to one-hot encoding")
        # Convert to one-hot encoding
        n_classes = int(y_train.max().item()) + 1
        y_train = F.one_hot(y_train.long(), num_classes=n_classes).float()
        y_valid = F.one_hot(y_valid.long(), num_classes=n_classes).float()

    X_valid = (X_valid - torch.mean(X_train, dim=0)) / \
        torch.std(X_train, dim=0)
    X_train = (X_train - torch.mean(X_train, dim=0)) / \
        torch.std(X_train, dim=0)

    log.info(
        f"After normalization - X_train: [{X_train.min():.3f}, {X_train.max():.3f}]")
    log.info(f"                     X_valid: [{
             X_valid.min():.3f}, {X_valid.max():.3f}]")
    log.info(
        f"Final shapes - X_train: {X_train.shape}, y_train: {y_train.shape}")
    log.info(f"              X_valid: {
             X_valid.shape}, y_valid: {y_valid.shape}")

    return X_train, y_train, X_valid, y_valid


class OriginalMNISTDataset:
    """Dataset wrapper for original MNIST data that implements AFADataset protocol"""

    # Class variables required by AFADataset protocol
    n_classes: int = 10
    n_features: int = 256

    def __init__(self, features: torch.Tensor, labels: torch.Tensor):
        self.features = features
        self.labels = labels
        # Update class variables based on actual data
        if hasattr(labels, 'shape') and len(labels.shape) > 1:
            self.n_classes = labels.shape[1]
        else:
            self.n_classes = int(labels.max().item()) + 1
        self.n_features = features.shape[1]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx: int):
        """Return a single sample from the dataset"""
        return self.features[idx], self.labels[idx]

    def get_all_data(self):
        """Return all data in the dataset"""
        return self.features, self.labels

    def generate_data(self):
        """Data is already generated, no-op"""
        pass

    def save(self, path: Path) -> None:
        """Save dataset to file"""
        torch.save({
            'features': self.features,
            'labels': self.labels
        }, path)

    @classmethod
    def load(cls, path: Path):
        """Load dataset from file"""
        data = torch.load(path)
        return cls(data['features'], data['labels'])


@hydra.main(
    version_base=None, config_path="../../conf/train/aco", config_name="config"
)
def main(cfg: AACOTrainConfig):
    log.debug(cfg)
    set_seed(cfg.seed)
    torch.set_float32_matmul_precision("medium")
    device = torch.device(cfg.device)

    run = wandb.init(
        config=cast(dict[str, Any], OmegaConf.to_container(cfg, resolve=True)),
        job_type="training",
        tags=["aaco"],
    )

    log.info(f"W&B run initialized: {run.name} ({run.id})")
    log.info(f"W&B run URL: {run.url}")

    # Check if we should use original MNIST data
    use_original_mnist = False

    if use_original_mnist:
        # Load original authors' MNIST data
        X_train, y_train, X_valid, y_valid = load_original_mnist_data()

        # Create dataset objects
        train_dataset = OriginalMNISTDataset(X_train, y_train)
        val_dataset = OriginalMNISTDataset(X_valid, y_valid)

        dataset_type = "MNIST_original"
        n_features = X_train.shape[-1]
        n_classes = y_train.shape[-1]

    else:
        # Use regular dataset loading
        train_dataset, val_dataset, _, dataset_metadata = load_dataset_artifact(
            cfg.dataset_artifact_name
        )

        dataset_type = dataset_metadata["dataset_type"]
        n_features = train_dataset.features.shape[-1]
        n_classes = train_dataset.labels.shape[-1]

        # Apply normalization for MNIST to fix k-NN density estimation
        if dataset_type.lower() == "mnist":
            log.info(
                "Applying normalization to MNIST data for better AACO performance...")

            X_train = train_dataset.features.float()
            X_valid = val_dataset.features.float()

            log.info(
                f"Normalized data range - Train: [{X_train.min():.3f}, {X_train.max():.3f}]")
            log.info(f"                       Valid: [{
                     X_valid.min():.3f}, {X_valid.max():.3f}]")

            # # Convert to float and normalize to [0,1]
            # X_train = train_dataset.features.float() / 255.0
            # X_valid = val_dataset.features.float() / 255.0

            # Update datasets with normalized data
            train_dataset.features = X_train
            val_dataset.features = X_valid

            log.info(
                f"Normalized data range - Train: [{X_train.min():.3f}, {X_train.max():.3f}]")
            log.info(f"                       Valid: [{
                     X_valid.min():.3f}, {X_valid.max():.3f}]")

    log.info(f"Dataset: {dataset_type}")
    log.info(f"Features: {n_features}, Classes: {n_classes}")
    log.info(f"Training samples: {len(train_dataset)}")

    aaco_method = create_aaco_method(
        k_neighbors=cfg.aco.k_neighbors,
        acquisition_cost=cfg.aco.acquisition_cost,
        hide_val=cfg.aco.hide_val,
        dataset_name=dataset_type.lower(),
        device=device,
    )

    log.info("Fitting AACO oracle on training data...")
    log.info(f"Using hyperparameters: k_neighbors={cfg.aco.k_neighbors}, acquisition_cost={
             cfg.aco.acquisition_cost}, hide_val={cfg.aco.hide_val}")

    X_train = train_dataset.features.to(device)
    y_train = train_dataset.labels.to(device)
    aaco_method.aaco_oracle.fit(X_train, y_train)

    log.info("AACO method created and fitted")

    run.log({
        "oracle_config/k_neighbors": cfg.aco.k_neighbors,
        "oracle_config/acquisition_cost": cfg.aco.acquisition_cost,
        "oracle_config/hide_val": cfg.aco.hide_val,
        "oracle_config/dataset_features": n_features,
        "oracle_config/dataset_classes": n_classes,
        "oracle_config/training_samples": len(train_dataset),
        "oracle_config/use_original_mnist": use_original_mnist,
    })

    if cfg.aco.evaluate_final_performance:
        log.info("Evaluating AACO method...")

        # budget = min(n_features, 25)

        metrics = eval_afa_method(
            afa_select_fn=aaco_method.select,
            dataset=val_dataset,
            # budget=budget,
            budget=None,  # AACO does not have a fixed budget
            afa_predict_fn=aaco_method.predict,
            only_n_samples=cfg.aco.eval_only_n_samples,
            device=device,
        )

        fig = plot_metrics(metrics)
        run.log({"aaco_metrics_plot": fig})

        final_accuracy = metrics["accuracy_all"][-1].item()
        final_f1 = metrics["f1_all"][-1].item()
        final_bce = metrics["bce_all"][-1].item()

        run.log({
            "final_accuracy": final_accuracy,
            "final_f1_score": final_f1,
            "final_bce": final_bce,
            "avg_features_used": len(metrics["accuracy_all"]),
        })

        log.info(f"Final performance - Accuracy: {final_accuracy:.4f}, F1: {
                 final_f1:.4f}, BCE: {final_bce:.4f}")

        budget_metrics = {}
        for i, (acc, f1, bce) in enumerate(zip(metrics["accuracy_all"], metrics["f1_all"], metrics["bce_all"])):
            budget_metrics[f"accuracy_at_{i+1}_features"] = acc.item()
            budget_metrics[f"f1_at_{i+1}_features"] = f1.item()
            budget_metrics[f"bce_at_{i+1}_features"] = bce.item()

        run.log(budget_metrics)

    with TemporaryDirectory(delete=False) as tmp_dir:
        tmp_path = Path(tmp_dir)

        aaco_method.save(tmp_path)

        aaco_artifact = wandb.Artifact(
            name=f"train_aaco-{dataset_type}-seed_{cfg.seed}",
            type="trained_method",
            metadata={
                "method_type": "aaco",
                "dataset_artifact_name": cfg.dataset_artifact_name if not use_original_mnist else "original_mnist",
                "dataset_type": dataset_type,
                "budget": None,  # AACO doesn't have a fixed budget
                "seed": cfg.seed,
                "k_neighbors": cfg.aco.k_neighbors,
                "acquisition_cost": cfg.aco.acquisition_cost,
                "hide_val": cfg.aco.hide_val,
                "implementation": "aaco",
                "use_original_mnist": use_original_mnist,
            },
        )
        aaco_artifact.add_dir(str(tmp_path))
        run.log_artifact(aaco_artifact, aliases=cfg.output_artifact_aliases)

    run.finish()


if __name__ == "__main__":
    main()
