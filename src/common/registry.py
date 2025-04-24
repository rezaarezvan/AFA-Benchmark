from afa_rl.afa_methods import (
    Shim2018AFAMethod,
    RandomDummyAFAMethod,
    SequentialDummyAFAMethod,
    Zannone2019AFAMethod,
)
from common.datasets import CubeDataset, AFAContextDataset, MNISTDataset, DiabetesDataset, PhysionetDataset

# Add each AFA method here
AFA_METHOD_REGISTRY = {
    "shim2018": Shim2018AFAMethod,
    "zannone2019": Zannone2019AFAMethod,
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
        "data/cube/train_split_1.pt",
        "data/cube/train_split_2.pt",
        "data/cube/train_split_3.pt",
        "data/cube/train_split_4.pt",
        "data/cube/train_split_5.pt",
        "data/cube/val_split_1.pt",
        "data/cube/val_split_2.pt",
        "data/cube/val_split_3.pt",
        "data/cube/val_split_4.pt",
        "data/cube/val_split_5.pt",
        "data/cube/test_split_1.pt",
        "data/cube/test_split_2.pt",
        "data/cube/test_split_3.pt",
        "data/cube/test_split_4.pt",
        "data/cube/test_split_5.pt",
    ]
}

# Add saved AFA methods here
TRAINING_REGISTRY = {
    # sequential_dummy
    (
        "sequential_dummy",
        "data/cube/train_split_1.pt",
    ): "models/sequential_dummy/sequential_dummy-cube_train_split_1.pt",
    (
        "sequential_dummy",
        "data/cube/train_split_2.pt",
    ): "models/sequential_dummy/sequential_dummy-cube_train_split_2.pt",
    (
        "sequential_dummy",
        "data/cube/train_split_3.pt",
    ): "models/sequential_dummy/sequential_dummy-cube_train_split_3.pt",
    (
        "sequential_dummy",
        "data/cube/train_split_4.pt",
    ): "models/sequential_dummy/sequential_dummy-cube_train_split_4.pt",
    (
        "sequential_dummy",
        "data/cube/train_split_5.pt",
    ): "models/sequential_dummy/sequential_dummy-cube_train_split_5.pt",
    # random_dummy
    (
        "random_dummy",
        "data/cube/train_split_1.pt",
    ): "models/random_dummy/random_dummy-cube_train_split_1.pt",
    (
        "random_dummy",
        "data/cube/train_split_2.pt",
    ): "models/random_dummy/random_dummy-cube_train_split_2.pt",
    (
        "random_dummy",
        "data/cube/train_split_3.pt",
    ): "models/random_dummy/random_dummy-cube_train_split_3.pt",
    (
        "random_dummy",
        "data/cube/train_split_4.pt",
    ): "models/random_dummy/random_dummy-cube_train_split_4.pt",
    (
        "random_dummy",
        "data/cube/train_split_5.pt",
    ): "models/random_dummy/random_dummy-cube_train_split_5.pt",
    # shim2018
    (
        "shim2018",
        "data/cube/train_split_1.pt",
    ): "models/shim2018/shim2018-cube_train_split_1.pt",
}

# Add evaluation results here
EVALUATION_REGISTRY = {
    # sequential_dummy
    "datasets": {
        "cube": {
            "methods": {
                "sequential_dummy": [
                    "results/evaluation/sequential_dummy/sequential_dummy-cube_train_split_1-cube_val_split_1.pt",
                    "results/evaluation/sequential_dummy/sequential_dummy-cube_train_split_2-cube_val_split_2.pt",
                    "results/evaluation/sequential_dummy/sequential_dummy-cube_train_split_3-cube_val_split_3.pt",
                    "results/evaluation/sequential_dummy/sequential_dummy-cube_train_split_4-cube_val_split_4.pt",
                    "results/evaluation/sequential_dummy/sequential_dummy-cube_train_split_5-cube_val_split_5.pt",
                ],
                "random_dummy": [
                    "results/evaluation/random_dummy/random_dummy-cube_train_split_1-cube_val_split_1.pt",
                    "results/evaluation/random_dummy/random_dummy-cube_train_split_2-cube_val_split_2.pt",
                    "results/evaluation/random_dummy/random_dummy-cube_train_split_3-cube_val_split_3.pt",
                    "results/evaluation/random_dummy/random_dummy-cube_train_split_4-cube_val_split_4.pt",
                    "results/evaluation/random_dummy/random_dummy-cube_train_split_5-cube_val_split_5.pt",
                ]
            }
        }
    }
}

