from afa_rl.afa_methods import (
    # Shim2018AFAMethod,
    RandomDummyAFAMethod,
    SequentialDummyAFAMethod,
    # Zannone2019AFAMethod,
    RLAFAMethod,
)
from afa_discriminative.afa_methods import (
    Covert2023AFAMethod,
    Gadgil2023AFAMethod,
)
from afa_generative.afa_methods import Ma2018AFAMethod
from common.custom_types import AFAClassifier, AFADataset, AFAMethod, PretrainingFunction, TrainingFunction
from common.datasets import CubeDataset, AFAContextDataset, MNISTDataset, DiabetesDataset, PhysionetDataset, Shim2018CubeDataset, MiniBooNEDataset, FashionMNISTDataset
from common.classifiers import (
    RandomDummyAFAClassifier,
    UniformDummyAFAClassifier,
)

from afa_rl.shim2018.scripts.pretrain_shim2018 import main as pretrain_shim2018_main
from afa_rl.shim2018.scripts.train_shim2018 import main as train_shim2018_main
from afa_rl.zannone2019.scripts.pretrain_zannone2019 import main as pretrain_zannone2019_main
from afa_rl.zannone2019.scripts.train_zannone2019_old import main as train_zannone2019_main


# Add each AFA method here
AFA_METHOD_REGISTRY: dict[str, type[AFAMethod]] = {
    "shim2018": RLAFAMethod,
    "zannone2019": RLAFAMethod,
    "covert2023": Covert2023AFAMethod,
    "gadgil2023": Gadgil2023AFAMethod,
    "ma2018": Ma2018AFAMethod,
    "sequential_dummy": SequentialDummyAFAMethod,  # For testing
    "random_dummy": RandomDummyAFAMethod,  # For testing
}

STATIC_METHOD_REGISTRY: list[str] = [
    "cae", 
    "permutation",
]

# Add each AFA dataset here
AFA_DATASET_REGISTRY: dict[str, type[AFADataset]] = {
    "cube": CubeDataset,
    "shim2018cube": Shim2018CubeDataset,
    "AFAContext": AFAContextDataset,
    "MNIST": MNISTDataset,
    "diabetes": DiabetesDataset,
    "physionet": PhysionetDataset,
    "miniboone": MiniBooNEDataset,
    "FashionMNIST": FashionMNISTDataset
}

# Add each common classifier that you want to use during evaluation here
AFA_CLASSIFIER_REGISTRY: dict[str, type[AFAClassifier]] = {
    "random_dummy": RandomDummyAFAClassifier,
    "uniform_dummy": UniformDummyAFAClassifier,
}

# Keep these last to avoid circular imports



PRETRAINING_ENTRY_REGISTRY: dict[str, PretrainingFunction] = {
    "shim2018": pretrain_shim2018_main,
    "zannone2019": pretrain_zannone2019_main,
}

TRAINING_ENTRY_REGISTRY: dict[str, TrainingFunction] = {
    "shim2018": train_shim2018_main,
    "zannone2019": train_zannone2019_main,
}
