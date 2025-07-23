from common.custom_types import (
    AFAMethod,
    AFADataset,
    AFAClassifier,
)

AFA_METHOD_TYPES = {
    "shim2018",
    "zannone2019",
    "kachuee2019",
    "covert2023",
    "gadgil2023",
    "ma2018",
    "aaco",
    "sequential_dummy",
    "random_dummy",
}


def get_afa_method_class(name: str) -> type[AFAMethod]:
    """Return the appropriate AFAMethod for a given method type.

    Note that several method types can have the same AFAMethod class, like the RL methods. A dictionary is not used since it could lead to circular imports."""

    if name in {"shim2018", "zannone2019", "kachuee2019"}:
        from afa_rl.afa_methods import RLAFAMethod

        return RLAFAMethod
    elif name == "covert2023":
        from afa_discriminative.afa_methods import Covert2023AFAMethod

        return Covert2023AFAMethod
    elif name == "gadgil2023":
        from afa_discriminative.afa_methods import Gadgil2023AFAMethod

        return Gadgil2023AFAMethod
    elif name == "ma2018":
        from afa_generative.afa_methods import Ma2018AFAMethod

        return Ma2018AFAMethod

    elif name == "aaco":
        from afa_oracle.afa_methods import AACOAFAMethod
        return AACOAFAMethod

    elif name == "sequentialdummy":
        from common.afa_methods import SequentialDummyAFAMethod

        return SequentialDummyAFAMethod
    elif name == "randomdummy":
        from common.afa_methods import RandomDummyAFAMethod

        return RandomDummyAFAMethod
    else:
        raise ValueError(f"Unknown AFA method: {name}")


AFA_DATASET_TYPES = {
    "cube",
    "shim2018cube",
    "AFAContext",
    "MNIST",
    "diabetes",
    "physionet",
    "miniboone",
    "FashionMNIST",
}


def get_afa_dataset_class(name: str) -> type[AFADataset]:
    if name == "cube":
        from common.datasets import CubeDataset

        return CubeDataset
    elif name == "shim2018cube":
        from common.datasets import Shim2018CubeDataset

        return Shim2018CubeDataset
    elif name == "AFAContext":
        from common.datasets import AFAContextDataset

        return AFAContextDataset
    elif name == "MNIST":
        from common.datasets import MNISTDataset

        return MNISTDataset
    elif name == "diabetes":
        from common.datasets import DiabetesDataset

        return DiabetesDataset
    elif name == "physionet":
        from common.datasets import PhysionetDataset

        return PhysionetDataset
    elif name == "miniboone":
        from common.datasets import MiniBooNEDataset

        return MiniBooNEDataset
    elif name == "FashionMNIST":
        from common.datasets import FashionMNISTDataset

        return FashionMNISTDataset
    else:
        raise ValueError(f"Unknown AFA dataset: {name}")


AFA_CLASSIFIER_TYPES = {
    "randomdummy",
    "uniformdummy",
    "WrappedMaskedMLPClassifier",
    "WrappedXGBoostAFAClassifier",
    "Shim2018AFAClassifier",
}


def get_afa_classifier_class(name: str) -> type[AFAClassifier]:
    if name == "randomdummy":
        from common.classifiers import RandomDummyAFAClassifier

        return RandomDummyAFAClassifier
    elif name == "uniformdummy":
        from common.classifiers import UniformDummyAFAClassifier

        return UniformDummyAFAClassifier
    elif name == "WrappedMaskedMLPClassifier":
        from common.classifiers import WrappedMaskedMLPClassifier

        return WrappedMaskedMLPClassifier
    elif name == "WrappedXGBoostAFAClassifier":
        from common.xgboost_classifier import WrappedXGBoostAFAClassifier

        return WrappedXGBoostAFAClassifier
    elif name == "Shim2018AFAClassifier":
        from afa_rl.shim2018.models import Shim2018AFAClassifier

        return Shim2018AFAClassifier
    elif name == "Zannone2019AFAClassifier":
        from afa_rl.zannone2019.models import Zannone2019AFAClassifier

        return Zannone2019AFAClassifier
    elif name == "Kachuee2019AFAClassifier":
        from afa_rl.kachuee2019.models import Kachuee2019AFAClassifier

        return Kachuee2019AFAClassifier
    else:
        raise ValueError(f"Unknown AFA classifier: {name}")
