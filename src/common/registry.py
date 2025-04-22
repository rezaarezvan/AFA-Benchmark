from afa_rl.afa_methods import (
    Shim2018AFAMethod,
    RandomDummyAFAMethod,
    SequentialDummyAFAMethod,
)
from common.datasets import CubeDataset, AFAContextDataset, MNISTDataset, DiabetesDataset, PhysionetDataset

# Add each AFA method here
AFA_METHOD_REGISTRY = {
    "shim2018": Shim2018AFAMethod,
    "sequential_dummy": SequentialDummyAFAMethod,  # For testing
    "random_dummy": RandomDummyAFAMethod,  # For testing
}

# Add each AFA dataset here
AFA_DATASET_REGISTRY = {
    "cube": CubeDataset,
    "AFAContext": AFAContextDataset,
    "MNIST": MNISTDataset,
    "diabetes": DiabetesDataset,
    "physionet": PhysionetDataset
}

# Mapping of AFA dataset names to their paths
AFA_DATASET_PATH_REGISTRY = {
    "cube": [
        "data/cube/cube_train_100.pt",
        "data/cube/cube_train_1000.pt",
        "data/cube/cube_train_10000.pt",
        "data/cube/cube_val_100.pt",
        "data/cube/cube_val_1000.pt",
    ]
}

# Add saved AFA methods here
TRAINING_REGISTRY = {
    (
        "sequential_dummy",
        "data/cube/cube_train_10000.pt",
    ): "models/afa_rl/sequential_dummy-cube_train_10000.pt",
    (
        "random_dummy",
        "data/cube/cube_train_10000.pt",
    ): "models/afa_rl/random_dummy-cube_train_10000.pt",
}

# Add evaluation results here
EVALUATION_REGISTRY = {
    # SequentialDummyAFAMethod
    (
        "models/afa_rl/sequential_dummy-cube_train_10000.pt",
        "data/cube/cube_val_1000.pt",
    ): "results/evaluation/sequential_dummy-cube_train_10000-cube_val_1000.pt",
    (
        "models/afa_rl/sequential_dummy-cube_train_10000.pt",
        "data/cube/cube_val_100.pt",
    ): "results/evaluation/sequential_dummy-cube_train_10000-cube_val_100.pt",
    # RandomDummyAFAMethod
    (
        "models/afa_rl/random_dummy-cube_train_10000.pt",
        "data/cube/cube_val_1000.pt",
    ): "results/evaluation/random_dummy-cube_train_10000-cube_val_1000.pt",
    (
        "models/afa_rl/random_dummy-cube_train_10000.pt",
        "data/cube/cube_val_100.pt",
    ): "results/evaluation/random_dummy-cube_train_10000-cube_val_100.pt",
}
