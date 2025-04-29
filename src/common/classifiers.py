
import torch
from common.custom_types import AFAClassifier, FeatureMask, Logits, MaskedFeatures


class RandomDummyAFAClassifier(AFAClassifier):
    """
    A random dummy classifier that outputs random logits. It is used for testing purposes.
    """

    def __init__(self, n_classes: int):
        self.n_classes = n_classes

    def __call__(self, masked_features: MaskedFeatures, feature_mask: FeatureMask) -> Logits:
        # Return random logits with the same batch size as masked_features
        batch_size = masked_features.shape[0]
        logits = torch.randn(batch_size, self.n_classes)

        return logits

    def save(self, path: str) -> None:
        """
        Saves the classifier to a file. n_classes is all we need.
        """
        torch.save(self.n_classes, path)

    @staticmethod
    def load(path: str, device: torch.device) -> "RandomDummyAFAClassifier":
        """
        Loads the classifier from a file, placing it on the given device.
        """
        # Load the number of classes
        n_classes = torch.load(path, map_location=device)

        # Return a new DummyClassifier instance
        return RandomDummyAFAClassifier(n_classes)


class UniformDummyAFAClassifier(AFAClassifier):
    """
    A uniform dummy classifier that outputs uniform logits. It is used for testing purposes.
    """

    def __init__(self, n_classes: int):
        self.n_classes = n_classes

    def __call__(self, masked_features: MaskedFeatures, feature_mask: FeatureMask) -> Logits:
        # Return random logits with the same batch size as masked_features
        batch_size = masked_features.shape[0]
        logits = torch.ones(batch_size, self.n_classes)

        return logits

    def save(self, path: str) -> None:
        """
        Saves the classifier to a file. n_classes is all we need.
        """
        torch.save(self.n_classes, path)

    @staticmethod
    def load(path: str, device: torch.device) -> "UniformDummyAFAClassifier":
        """
        Loads the classifier from a file, placing it on the given device.
        """
        # Load the number of classes
        n_classes = torch.load(path, map_location=device)

        # Return a new DummyClassifier instance
        return UniformDummyAFAClassifier(n_classes)
