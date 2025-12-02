from afabench.common.config_classes import (
    AACODefaultInitializerConfig,
    FixedRandomInitializerConfig,
    ImagePatchUnmaskerConfig,
    InitializerConfig,
    LeastInformativeInitializerConfig,
    ManualInitializerConfig,
    MutualInformationInitializerConfig,
    RandomPerEpisodeInitializerConfig,
    UnmaskerConfig,
)
from afabench.common.custom_types import (
    AFAClassifier,
    AFADataset,
    AFAInitializer,
    AFAMethod,
    AFAUnmasker,
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
    if name == "RLAFAMethod":
        # Lazy import of RLAFAMethod moved inside the function to avoid circular dependency
        from afabench.afa_rl.afa_methods import RLAFAMethod

        return RLAFAMethod
    if name == "Covert2023AFAMethod":
        from afabench.afa_discriminative.afa_methods import (
            Covert2023AFAMethod,
        )

        return Covert2023AFAMethod
    if name == "Gadgil2023AFAMethod":
        from afabench.afa_discriminative.afa_methods import (
            Gadgil2023AFAMethod,
        )

        return Gadgil2023AFAMethod
    if name == "Ma2018AFAMethod":
        from afabench.afa_generative.afa_methods import (
            Ma2018AFAMethod,
        )

        return Ma2018AFAMethod

    if name == "AACOAFAMethod":
        from afabench.afa_oracle.afa_methods import (
            AACOAFAMethod,
        )

        return AACOAFAMethod

    if name == "StaticBaseMethod":
        from afabench.static.static_methods import (
            StaticBaseMethod,
        )

        return StaticBaseMethod
    if name == "SequentialDummyAFAMethod":
        from afabench.common.afa_methods import (
            SequentialDummyAFAMethod,
        )

        return SequentialDummyAFAMethod
    if name == "RandomDummyAFAMethod":
        from afabench.common.afa_methods import (
            RandomDummyAFAMethod,
        )

        return RandomDummyAFAMethod
    if name == "OptimalCubeAFAMethod":
        from afabench.common.afa_methods import (
            OptimalCubeAFAMethod,
        )

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
        from afabench.common.datasets.datasets import (
            CubeDataset,
        )

        return CubeDataset
    if name == "AFAContext":
        from afabench.common.datasets.datasets import (
            AFAContextDataset,
        )

        return AFAContextDataset
    if name == "MNIST":
        from afabench.common.datasets.datasets import (
            MNISTDataset,
        )

        return MNISTDataset
    if name == "diabetes":
        from afabench.common.datasets.datasets import (
            DiabetesDataset,
        )

        return DiabetesDataset
    if name == "physionet":
        from afabench.common.datasets.datasets import (
            PhysionetDataset,
        )

        return PhysionetDataset
    if name == "miniboone":
        from afabench.common.datasets.datasets import (
            MiniBooNEDataset,
        )

        return MiniBooNEDataset
    if name == "FashionMNIST":
        from afabench.common.datasets.datasets import (
            FashionMNISTDataset,
        )

        return FashionMNISTDataset
    if name == "bank_marketing":
        from afabench.common.datasets.datasets import BankMarketingDataset

        return BankMarketingDataset

    if name == "ckd":
        from afabench.common.datasets.datasets import CKDDataset

        return CKDDataset

    if name == "actg":
        from afabench.common.datasets.datasets import ACTG175Dataset

        return ACTG175Dataset

    if name == "imagenette":
        from afabench.common.datasets.datasets import ImagenetteDataset

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


def get_afa_classifier_class(name: str) -> type[AFAClassifier]:  # noqa: PLR0911
    if name == "randomdummy":
        from afabench.common.classifiers import (
            RandomDummyAFAClassifier,
        )

        return RandomDummyAFAClassifier
    if name == "uniformdummy":
        from afabench.common.classifiers import (
            UniformDummyAFAClassifier,
        )

        return UniformDummyAFAClassifier
    if name == "WrappedMaskedMLPClassifier":
        from afabench.common.classifiers import (
            WrappedMaskedMLPClassifier,
        )

        return WrappedMaskedMLPClassifier

    if name == "WrappedMaskedViTClassifier":
        from afabench.common.classifiers import (
            WrappedMaskedViTClassifier,
        )

        return WrappedMaskedViTClassifier
    if name == "Shim2018AFAClassifier":
        from afabench.afa_rl.shim2018.models import (
            Shim2018AFAClassifier,
        )

        return Shim2018AFAClassifier
    if name == "Zannone2019AFAClassifier":
        from afabench.afa_rl.zannone2019.models import (
            Zannone2019AFAClassifier,
        )

        return Zannone2019AFAClassifier
    if name == "Kachuee2019AFAClassifier":
        from afabench.afa_rl.kachuee2019.models import (
            Kachuee2019AFAClassifier,
        )

        return Kachuee2019AFAClassifier
    msg = f"Unknown AFA classifier: {name}"
    raise ValueError(msg)


AFA_UNMASKER_TYPES = {"one_based_index", "image_patch"}


def get_afa_unmasker(unmasker_cfg: UnmaskerConfig) -> AFAUnmasker:
    """Get unmasker function by name."""
    if unmasker_cfg.type == "one_based_index":
        assert unmasker_cfg.config is None, (
            "one_based_index unmasker takes no config"
        )
        from afabench.common.afa_unmaskers import DirectUnmasker

        return DirectUnmasker()

    if unmasker_cfg.type == "image_patch":
        assert isinstance(unmasker_cfg.config, ImagePatchUnmaskerConfig), (
            "image_patch unmasker requires ImagePatchUnmaskerConfig"
        )
        from afabench.common.afa_unmaskers import ImagePatchUnmasker

        return ImagePatchUnmasker(config=unmasker_cfg.config)

    msg = f"Unknown unmasker: {unmasker_cfg.type}"
    raise ValueError(msg)


AFA_INITIALIZER_TYPES = {
    "zero",
    "fixed_random",
    "random_per_episode",
    "manual",
    "mutual_information",
    "least_informative",
    "aaco_default",
}


def get_afa_initializer(initializer_cfg: InitializerConfig) -> AFAInitializer:  # noqa: PLR0911
    """Get initializer by name."""
    if initializer_cfg.type == "zero":
        # ZeroInitializer does not require a config object
        assert initializer_cfg.config is None, (
            "zero initializer must not have a config object"
        )
        from afabench.common.afa_initializers import ZeroInitializer

        return ZeroInitializer()

    if initializer_cfg.type == "fixed_random":
        assert isinstance(
            initializer_cfg.config, FixedRandomInitializerConfig
        ), "fixed_random initializer requires FixedRandomInitializerConfig"
        from afabench.common.afa_initializers import FixedRandomInitializer

        return FixedRandomInitializer(config=initializer_cfg.config)

    if initializer_cfg.type == "random_per_episode":
        assert isinstance(
            initializer_cfg.config, RandomPerEpisodeInitializerConfig
        ), "expected ManualInitializerConfig"
        from afabench.common.afa_initializers import (
            RandomPerEpisodeInitializer,
        )

        return RandomPerEpisodeInitializer(config=initializer_cfg.config)

    if initializer_cfg.type == "manual":
        assert isinstance(initializer_cfg.config, ManualInitializerConfig), (
            "expected ManualInitializerConfig"
        )
        from afabench.common.afa_initializers import ManualInitializer

        return ManualInitializer(config=initializer_cfg.config)

    if initializer_cfg.type == "mutual_information":
        assert isinstance(
            initializer_cfg.config, MutualInformationInitializerConfig
        ), (
            "mutual_information initializer requires MutualInformationInitializerConfig"
        )
        from afabench.common.afa_initializers import (
            MutualInformationInitializer,
        )

        return MutualInformationInitializer(config=initializer_cfg.config)

    if initializer_cfg.type == "least_informative":
        assert isinstance(
            initializer_cfg.config, LeastInformativeInitializerConfig
        ), (
            "least_informative initializer requires LeastInformativeInitializerConfig"
        )
        from afabench.common.afa_initializers import (
            LeastInformativeInitializer,
        )

        return LeastInformativeInitializer(config=initializer_cfg.config)

    if initializer_cfg.type == "aaco_default":
        assert isinstance(
            initializer_cfg.config, AACODefaultInitializerConfig
        ), "expected AACODefaultInitializerConfig"
        from afabench.common.afa_initializers import (
            AACODefaultInitializer,
        )

        return AACODefaultInitializer(config=initializer_cfg.config)

    msg = f"Unknown initializer: {initializer_cfg.type}"
    raise ValueError(msg)
