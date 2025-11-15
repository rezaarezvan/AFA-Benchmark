from common.custom_types import (
    AFAClassifier,
    AFADataset,
    AFAMethod,
)

AFA_METHOD_TYPES = {
    "shim2018",
    "zannone2019",
    "kachuee2019",
    "covert2023",
    "gadgil2023",
    "ma2018",
    "aaco",
    "cae",
    "permutation",
    "sequential_dummy",
    "random_dummy",
    "optimalcube",
}


def get_afa_method_class(name: str) -> type[AFAMethod]:  # noqa: PLR0911
    """
    Return the appropriate AFAMethod for a given method type.

    Note that several method types can have the same AFAMethod class, like the RL methods. A dictionary is not used since it could lead to circular imports.
    """
    if name in {"shim2018", "zannone2019", "kachuee2019"}:
        from afa_rl.afa_methods import RLAFAMethod  # noqa: PLC0415

        return RLAFAMethod
    if name == "covert2023":
        from afa_discriminative.afa_methods import (  # noqa: PLC0415
            Covert2023AFAMethod,
        )

        return Covert2023AFAMethod
    if name == "gadgil2023":
        from afa_discriminative.afa_methods import (  # noqa: PLC0415
            Gadgil2023AFAMethod,
        )

        return Gadgil2023AFAMethod
    if name == "ma2018":
        from afa_generative.afa_methods import Ma2018AFAMethod  # noqa: PLC0415

        return Ma2018AFAMethod

    if name == "aaco":
        from afa_oracle.afa_methods import AACOAFAMethod  # noqa: PLC0415

        return AACOAFAMethod

    if name in {"cae", "permutation"}:
        from static.static_methods import StaticBaseMethod  # noqa: PLC0415

        return StaticBaseMethod
    if name == "sequentialdummy":
        from common.afa_methods import (  # noqa: PLC0415
            SequentialDummyAFAMethod,
        )

        return SequentialDummyAFAMethod
    if name == "randomdummy":
        from common.afa_methods import RandomDummyAFAMethod  # noqa: PLC0415

        return RandomDummyAFAMethod
    if name == "optimalcube":
        from common.afa_methods import OptimalCubeAFAMethod  # noqa: PLC0415

        return OptimalCubeAFAMethod
    msg = f"Unknown AFA method: {name}"
    raise ValueError(msg)


AFA_DATASET_TYPES = {
    "cube",
    "cubeSimple",
    "cubeOnlyInformative",
    "shim2018cube",
    "AFAContext",
    "AFAContextRandomInsert",
    "ContextSelectiveXOR",
    "MNIST",
    "diabetes",
    "physionet",
    "miniboone",
    "FashionMNIST",
    "bank_marketing",
    "ckd",
    "actg",
}


def get_afa_dataset_class(name: str) -> type[AFADataset]:  # noqa: C901, PLR0911
    if name == "cube":
        from common.datasets import CubeDataset  # noqa: PLC0415

        return CubeDataset
    if name == "cubeSimple":
        from common.datasets import CubeSimpleDataset  # noqa: PLC0415

        return CubeSimpleDataset
    if name == "cubeOnlyInformative":
        from common.datasets import CubeOnlyInformativeDataset  # noqa: PLC0415

        return CubeOnlyInformativeDataset
    if name == "shim2018cube":
        from common.datasets import Shim2018CubeDataset  # noqa: PLC0415

        return Shim2018CubeDataset
    if name == "AFAContext":
        from common.datasets import AFAContextDataset  # noqa: PLC0415

        return AFAContextDataset
    if name == "AFAContextRandomInsert":
        from common.datasets import (  # noqa: PLC0415
            AFAContextRandomInsertDataset,
        )

        return AFAContextRandomInsertDataset
    if name == "ContextSelectiveXOR":
        from common.datasets import ContextSelectiveXORDataset  # noqa: PLC0415

        return ContextSelectiveXORDataset
    if name == "MNIST":
        from common.datasets import MNISTDataset  # noqa: PLC0415

        return MNISTDataset
    if name == "diabetes":
        from common.datasets import DiabetesDataset  # noqa: PLC0415

        return DiabetesDataset
    if name == "physionet":
        from common.datasets import PhysionetDataset  # noqa: PLC0415

        return PhysionetDataset
    if name == "miniboone":
        from common.datasets import MiniBooNEDataset  # noqa: PLC0415

        return MiniBooNEDataset
    if name == "FashionMNIST":
        from common.datasets import FashionMNISTDataset  # noqa: PLC0415

        return FashionMNISTDataset
    if name == "bank_marketing":
        from common.datasets import BankMarketingDataset

        return BankMarketingDataset

    if name == "ckd":
        from common.datasets import CKDDataset

        return CKDDataset

    if name == "actg":
        from common.datasets import ACTG175Dataset

        return ACTG175Dataset

    if name == "imagenette":
        from common.datasets import ImagenetteDataset

        return ImagenetteDataset

    msg = f"Unknown AFA dataset: {name}"
    raise ValueError(msg)


AFA_CLASSIFIER_TYPES = {
    "randomdummy",
    "uniformdummy",
    "WrappedMaskedMLPClassifier",
    "WrappedMaskedVitClassifier",
    "Shim2018AFAClassifier",
}


def get_afa_classifier_class(name: str) -> type[AFAClassifier]:
    if name == "randomdummy":
        from common.classifiers import (  # noqa: PLC0415
            RandomDummyAFAClassifier,
        )

        return RandomDummyAFAClassifier
    if name == "uniformdummy":
        from common.classifiers import (  # noqa: PLC0415
            UniformDummyAFAClassifier,
        )

        return UniformDummyAFAClassifier
    if name == "WrappedMaskedMLPClassifier":
        from common.classifiers import (  # noqa: PLC0415
            WrappedMaskedMLPClassifier,
        )

        return WrappedMaskedMLPClassifier

    if name == "WrappedMaskedVitClassifier":
        from common.classifiers import (
            WrappedMaskedViTClassifier,
        )

        return WrappedMaskedViTClassifier
    if name == "Shim2018AFAClassifier":
        from afa_rl.shim2018.models import (  # noqa: PLC0415
            Shim2018AFAClassifier,
        )

        return Shim2018AFAClassifier
    if name == "Zannone2019AFAClassifier":
        from afa_rl.zannone2019.models import (  # noqa: PLC0415
            Zannone2019AFAClassifier,
        )

        return Zannone2019AFAClassifier
    if name == "Kachuee2019AFAClassifier":
        from afa_rl.kachuee2019.models import (  # noqa: PLC0415
            Kachuee2019AFAClassifier,
        )

        return Kachuee2019AFAClassifier
    msg = f"Unknown AFA classifier: {name}"
    raise ValueError(msg)
