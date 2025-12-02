from afabench.common.custom_types import (
    AFAClassifier,
    AFADataset,
    AFAInitializer,
    AFAMethod,
    AFAUnmasker,
)

AFA_METHOD_CLASSES = {
    "RLAFAMethod": "afabench.afa_rl.afa_methods.RLAFAMethod",
    "Covert2023AFAMethod": "afabench.afa_discriminative.afa_methods.Covert2023AFAMethod",
    "Gadgil2023AFAMethod": "afabench.afa_discriminative.afa_methods.Gadgil2023AFAMethod",
    "Ma2018AFAMethod": "afabench.afa_generative.afa_methods.Ma2018AFAMethod",
    "AACOAFAMethod": "afabench.afa_oracle.afa_methods.AACOAFAMethod",
    "StaticBaseMethod": "afabench.static.static_methods.StaticBaseMethod",
    "SequentialDummyAFAMethod": "afabench.common.afa_methods.SequentialDummyAFAMethod",
    "RandomDummyAFAMethod": "afabench.common.afa_methods.RandomDummyAFAMethod",
    "OptimalCubeAFAMethod": "afabench.common.afa_methods.OptimalCubeAFAMethod",
}


def get_afa_method_class(class_name: str) -> type[AFAMethod]:
    """Return a reference to an AFAMethod class, given the class name."""
    if class_name not in AFA_METHOD_CLASSES:
        msg = f"Unknown AFA method: {class_name}"
        raise ValueError(msg)

    module_name, class_name = AFA_METHOD_CLASSES[class_name].rsplit(".", 1)
    module = __import__(module_name, fromlist=[class_name])
    return getattr(module, class_name)


AFA_DATASET_CLASSES = {
    "CubeDataset": "afabench.common.datasets.datasets.CubeDataset",
    "AFAContextDataset": "afabench.common.datasets.datasets.AFAContextDataset",
    "MNISTDataset": "afabench.common.datasets.datasets.MNISTDataset",
    "DiabetesDataset": "afabench.common.datasets.datasets.DiabetesDataset",
    "PhysionetDataset": "afabench.common.datasets.datasets.PhysionetDataset",
    "MiniBooNEDataset": "afabench.common.datasets.datasets.MiniBooNEDataset",
    "FashionMNISTDataset": "afabench.common.datasets.datasets.FashionMNISTDataset",
    "BankMarketingDataset": "afabench.common.datasets.datasets.BankMarketingDataset",
    "CKDDataset": "afabench.common.datasets.datasets.CKDDataset",
    "ACTG175Dataset": "afabench.common.datasets.datasets.ACTG175Dataset",
    "ImagenetteDataset": "afabench.common.datasets.datasets.ImagenetteDataset",
}


def get_afa_dataset_class(class_name: str) -> type[AFADataset]:
    """Return a reference to an AFADataset class, given the class name."""
    if class_name not in AFA_DATASET_CLASSES:
        msg = f"Unknown AFA dataset: {class_name}"
        raise ValueError(msg)

    module_name, class_name = AFA_DATASET_CLASSES[class_name].rsplit(".", 1)
    module = __import__(module_name, fromlist=[class_name])
    return getattr(module, class_name)


AFA_CLASSIFIER_CLASSES = {
    "randomdummy": "afabench.common.classifiers.RandomDummyAFAClassifier",
    "uniformdummy": "afabench.common.classifiers.UniformDummyAFAClassifier",
    "WrappedMaskedMLPClassifier": "afabench.common.classifiers.WrappedMaskedMLPClassifier",
    "WrappedMaskedViTClassifier": "afabench.common.classifiers.WrappedMaskedViTClassifier",
    "Shim2018AFAClassifier": "afabench.afa_rl.shim2018.models.Shim2018AFAClassifier",
    "Zannone2019AFAClassifier": "afabench.afa_rl.zannone2019.models.Zannone2019AFAClassifier",
    "Kachuee2019AFAClassifier": "afabench.afa_rl.kachuee2019.models.Kachuee2019AFAClassifier",
}


def get_afa_classifier_class(class_name: str) -> type[AFAClassifier]:
    """Return a reference to an AFAClassifier class, given the class name."""
    if class_name not in AFA_CLASSIFIER_CLASSES:
        msg = f"Unknown AFA classifier: {class_name}"
        raise ValueError(msg)

    module_name, class_name = AFA_CLASSIFIER_CLASSES[class_name].rsplit(".", 1)
    module = __import__(module_name, fromlist=[class_name])
    return getattr(module, class_name)


AFA_UNMASKER_CLASSES = {
    "DirectUnmasker": "afabench.common.afa_unmaskers.DirectUnmasker",
    "ImagePatchUnmasker": "afabench.common.afa_unmaskers.ImagePatchUnmasker",
}


def get_afa_unmasker_class(class_name: str) -> type[AFAUnmasker]:
    """Return a reference to an AFAUnmasker class, given the class name."""
    if class_name not in AFA_UNMASKER_CLASSES:
        msg = f"Unknown unmasker: {class_name}"
        raise ValueError(msg)

    module_name, class_name = AFA_UNMASKER_CLASSES[class_name].rsplit(".", 1)
    module = __import__(module_name, fromlist=[class_name])
    return getattr(module, class_name)


AFA_INITIALIZER_CLASSES = {
    "ZeroInitializer": "afabench.common.afa_initializers.ZeroInitializer",
    "FixedRandomInitializer": "afabench.common.afa_initializers.FixedRandomInitializer",
    "DynamicRandomInitializer": "afabench.common.afa_initializers.DynamicRandomInitializer",
    "ManualInitializer": "afabench.common.afa_initializers.ManualInitializer",
    "MutualInformationInitializer": "afabench.common.afa_initializers.MutualInformationInitializer",
    "LeastInformativeInitializer": "afabench.common.afa_initializers.LeastInformativeInitializer",
    "AACODefaultInitializer": "afabench.common.afa_initializers.AACODefaultInitializer",
}


def get_afa_initializer_class(class_name: str) -> type[AFAInitializer]:
    """Return a reference to an AFAInitializer class, given the class name."""
    if class_name not in AFA_INITIALIZER_CLASSES:
        msg = f"Unknown initializer: {class_name}"
        raise ValueError(msg)

    module_name, class_name = AFA_INITIALIZER_CLASSES[class_name].rsplit(
        ".", 1
    )
    module = __import__(module_name, fromlist=[class_name])
    return getattr(module, class_name)
