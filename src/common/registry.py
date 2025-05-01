from afa_rl.afa_methods import (
    Shim2018AFAMethod,
    RandomDummyAFAMethod,
    SequentialDummyAFAMethod,
    Zannone2019AFAMethod,
)
from afa_discriminative.afa_methods import (
    Covert2023AFAMethod,
    Gadgil2023AFAMethod,
)
from common.custom_types import AFAClassifier, AFADataset, AFAMethod, PretrainingFunction, TrainingFunction
from common.datasets import CubeDataset, AFAContextDataset, MNISTDataset, DiabetesDataset, PhysionetDataset
from common.classifiers import (
    RandomDummyAFAClassifier,
    UniformDummyAFAClassifier,
)

from afa_rl.scripts.pretrain_shim2018 import main as pretrain_shim2018_main
from afa_rl.scripts.train_shim2018 import main as train_shim2018_main
from afa_rl.scripts.pretrain_zannone2019 import main as pretrain_zannone2019_main
from afa_rl.scripts.train_zannone2019 import main as train_zannone2019_main


# Add each AFA method here
AFA_METHOD_REGISTRY: dict[str, type[AFAMethod]] = {
    "shim2018": Shim2018AFAMethod,
    "zannone2019": Zannone2019AFAMethod,
    "GDFS": Covert2023AFAMethod,
    "DIME": Gadgil2023AFAMethod,
    "sequential_dummy": SequentialDummyAFAMethod,  # For testing
    "random_dummy": RandomDummyAFAMethod,  # For testing
}



# Add each AFA dataset here
AFA_DATASET_REGISTRY: dict[str, type[AFADataset]] = {
    "cube": CubeDataset,
    "AFAContext": AFAContextDataset,
    "MNIST": MNISTDataset,
    "diabetes": DiabetesDataset,
    "physionet": PhysionetDataset
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
