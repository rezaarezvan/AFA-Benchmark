
import torch
import torch.nn as nn
from pathlib import Path
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
    

class Predictor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, x):
        return self.model(x)


class NNClassifier(AFAClassifier):
    """
    A trainable classifier that uses a simple predictor
    and handles masked input.
    """
    def __init__(self, input_dim: int, output_dim: int, device: torch.device):
        super().__init__()
        self.device = device
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.predictor = Predictor(input_dim, output_dim).to(device)

    def __call__(self, masked_features: MaskedFeatures, feature_mask: FeatureMask) -> Logits:
        x_masked = torch.cat([masked_features, feature_mask], dim=1)
        return self.predictor(x_masked)
    
    def save(self, path: Path) -> None:
        torch.save({
            "model_state_dict": self.predictor.state_dict(),
            "input_dim": self.input_dim,
            "output_dim": self.output_dim
        }, path)

    @staticmethod
    def load(path: Path, device: torch.device) -> "AFAClassifier":
        checkpoint = torch.load(path, map_location=device)
        classifier = NNClassifier(checkpoint["input_dim"], checkpoint["output_dim"], device)
        classifier.predictor.load_state_dict(checkpoint["model_state_dict"])
        return classifier
