from afa_rl.afa_methods import Shim2018AFAMethod
from common.datasets import CubeDataset

# Add each AFA method here
AFA_METHOD_REGISTRY = {"shim2018": Shim2018AFAMethod}

# Add each AFA dataset here
AFA_DATASET_REGISTRY = {"cube": CubeDataset}

# Add saved AFA methods here
TRAINING_REGISTRY = {
    ("shim2018", "cube_train_1.pt"): "shim2018-cube_train_1.pt",
}

# Add evaluation results here
EVALUATION_REGISTRY = {
    (
        "shim2018-cube_train_1.pt",
        "cube_val_1.pt",
    ): "shim2018-cube_train_1-cube_val_1.pt",
}
