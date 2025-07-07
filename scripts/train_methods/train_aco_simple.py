import torch
import logging

from pathlib import Path
from common.models import MaskedMLPClassifier
from afa_oracle.afa_methods import ACOOracleMethod
from common.datasets import CubeDataset, MNISTDataset

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def load_dataset_from_file(
    dataset_name: str, split: int = 1, split_type: str = "train"
):
    """
    Load dataset from .pt file in data/ directory.
    """
    dataset_path = Path(f"data/{dataset_name}/{split_type}_split_{split}.pt")

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    # Load based on dataset type
    if dataset_name.lower() == "cube":
        return CubeDataset.load(dataset_path)
    elif dataset_name.lower() == "mnist":
        return MNISTDataset.load(dataset_path)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def create_simple_classifier(n_features: int, n_classes: int, device: str = "cpu"):
    """
    Create and train a simple classifier for testing.
    """
    model = MaskedMLPClassifier(
        n_features=n_features,
        n_classes=n_classes,
        num_cells=(128, 128),  # tuple format required
        dropout=0.1,
    ).to(device)

    # For now, just return untrained model for structure testing
    log.warning("Using untrained classifier - only for structure testing!")
    return model


def main():
    # Simple configuration
    dataset_name = "cube"
    split = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log.info(f"Loading {dataset_name} dataset split {split}")

    # Load training dataset
    try:
        train_dataset = load_dataset_from_file(dataset_name, split, "train")
        log.info(f"Loaded train dataset: {len(train_dataset)} samples")
    except FileNotFoundError as e:
        log.error(f"Dataset loading failed: {e}")
        log.info("Available datasets in data/:")
        data_dir = Path("data")
        if data_dir.exists():
            for subdir in data_dir.iterdir():
                if subdir.is_dir():
                    log.info(f"  {subdir.name}/")
                    for file in subdir.iterdir():
                        log.info(f"    {file.name}")
        return

    # Get dataset properties
    n_features = train_dataset.features.shape[1]
    n_classes = train_dataset.labels.shape[1]  # One-hot encoded

    log.info(f"Dataset: {n_features} features, {n_classes} classes")

    # Get feature costs
    if hasattr(train_dataset, "costs"):
        costs = train_dataset.costs
        log.info(f"Feature costs: min={costs.min():.3f}, max={costs.max():.3f}")
    else:
        costs = torch.ones(n_features)
        log.info("Using uniform feature costs")

    # Create simple classifier
    classifier = create_simple_classifier(n_features, n_classes, device)
    log.info("Created classifier")

    # Create ACO method
    log.info("Creating ACO oracle...")
    aco_method = ACOOracleMethod(
        predictor=classifier,
        X_train=train_dataset.features,
        y_train=train_dataset.labels,  # Keep as one-hot, ACOOracleMethod will convert
        costs=costs,
        k=5,
        alpha=0.01,
        method="full",  # Start with greedy for simplicity
        device=device,
    )

    log.info("ACO oracle created successfully!")

    # Test basic functionality
    log.info("Testing ACO oracle...")

    # Create a small test batch
    test_batch = {
        "x": train_dataset.features[:2],  # First 2 samples
        "y": train_dataset.labels[:2],
    }

    # Test reset
    aco_method.reset(test_batch)
    log.info("Reset successful")

    # Test feature selection
    try:
        actions = aco_method.select_feature()
        log.info(f"Feature selection successful: {actions}")

        # Test prediction
        predictions = aco_method.get_predictions()
        log.info(f"Prediction successful: shape {predictions.shape}")

    except Exception as e:
        log.error(f"ACO oracle test failed: {e}")
        raise

    # Test saving/loading
    output_dir = Path("test_aco_output")
    output_dir.mkdir(exist_ok=True)

    log.info(f"Testing save/load to {output_dir}")
    aco_method.save(output_dir)
    log.info("Save successful")

    # Load and test
    loaded_method = ACOOracleMethod.load(output_dir, device)
    loaded_method.reset(test_batch)
    loaded_actions = loaded_method.select_feature()
    log.info(f"Load successful: {loaded_actions}")

    log.info("All tests passed! ACO oracle is ready.")


if __name__ == "__main__":
    main()
