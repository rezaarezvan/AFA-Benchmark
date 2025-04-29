from afa_rl.afa_methods import (
    Shim2018AFAMethod,
    RandomDummyAFAMethod,
    SequentialDummyAFAMethod,
    Zannone2019AFAMethod,
)
from afa_discriminative.afa_methods import (
    GreedyDynamicSelection,
    CMIEstimator,
)
from common.datasets import CubeDataset, AFAContextDataset, MNISTDataset, DiabetesDataset, PhysionetDataset

# Add each AFA method here
AFA_METHOD_REGISTRY = {
    "shim2018": Shim2018AFAMethod,
    "zannone2019": Zannone2019AFAMethod,
    "GDFS": GreedyDynamicSelection,
    "DIME": CMIEstimator,
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
