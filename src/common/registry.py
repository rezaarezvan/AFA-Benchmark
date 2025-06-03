from common.custom_types import (
    AFAMethod,
    AFADataset,
    AFAClassifier,
)


def get_afa_method_class(name: str) -> type[AFAMethod]:
    if name == "RLAFAMethod":
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
    elif name == "sequential_dummy":
        from afa_rl.afa_methods import SequentialDummyAFAMethod

        return SequentialDummyAFAMethod
    elif name == "random_dummy":
        from afa_rl.afa_methods import RandomDummyAFAMethod

        return RandomDummyAFAMethod
    else:
        raise ValueError(f"Unknown AFA method: {name}")


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


def get_afa_classifier_class(name: str) -> type[AFAClassifier]:
    if name == "random_dummy":
        from common.classifiers import RandomDummyAFAClassifier

        return RandomDummyAFAClassifier
    elif name == "uniform_dummy":
        from common.classifiers import UniformDummyAFAClassifier

        return UniformDummyAFAClassifier
    elif name == "WrappedMaskedMLPClassifier":
        from common.classifiers import WrappedMaskedMLPClassifier

        return WrappedMaskedMLPClassifier
    elif name == "Shim2018AFAClassifier":
        from afa_rl.shim2018.models import Shim2018AFAClassifier

        return Shim2018AFAClassifier
    else:
        raise ValueError(f"Unknown AFA classifier: {name}")
