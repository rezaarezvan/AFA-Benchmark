from pathlib import Path
from typing import Self, final, override
import torch

from afa_rl.utils import afacontext_optimal_selection
from common.custom_types import (
    AFAClassifier,
    AFAMethod,
    AFASelection,
    FeatureMask,
    Features,
    Label,
    MaskedFeatures,
)


@final
class RandomDummyAFAMethod(AFAMethod):
    """A dummy AFAMethod for testing purposes. Chooses a random feature to observe from the masked features."""

    def __init__(self, device: torch.device, n_classes: int):
        self._device = device
        self.n_classes = n_classes

    @override
    def select(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
        features: Features,
        label: Label,
    ) -> AFASelection:
        """Chooses to observe a random feature from the masked features (or stop collecting features)."""
        original_device = masked_features.device

        masked_features = masked_features.to(self._device)
        feature_mask = feature_mask.to(self._device)

        # Sample from unobserved features uniformly
        probs = (~feature_mask).float()

        # Avoid division by zero
        row_sums = probs.sum(dim=1, keepdim=True)
        probs = torch.where(
            row_sums > 0, probs / row_sums, probs
        )  # normalize or leave zeros

        # Sample one index per row
        sampled = torch.multinomial(probs, num_samples=1)
        selection = sampled.squeeze(1)  # (B, 1) → (B,)

        return selection.to(original_device)

    @override
    def predict(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
        features: Features,
        label: Label,
    ) -> Label:
        """Return a random prediction from the classes."""
        original_device = masked_features.device

        masked_features = masked_features.to(self._device)
        feature_mask = feature_mask.to(self._device)

        # Pick a random class from the classes
        prediction = torch.randint(
            0,
            self.n_classes,
            (masked_features.shape[0],),
            device=masked_features.device,
        )
        # One-hot encode the prediction
        prediction = torch.nn.functional.one_hot(
            prediction, num_classes=self.n_classes
        ).float()

        return prediction.to(original_device)

    @override
    def save(self, path: Path) -> None:
        """Save the method to a folder."""
        torch.save(
            {
                "n_classes": self.n_classes,
            },
            path / "method.pt",
        )

    @classmethod
    @override
    def load(cls, path: Path, device: torch.device) -> Self:
        """Load the method from a file."""
        data = torch.load(path / "method.pt")
        return cls(device, data["n_classes"])

    @override
    def to(self, device: torch.device) -> Self:
        self._device = device
        return self

    @property
    @override
    def device(self) -> torch.device:
        return self._device


@final
class SequentialDummyAFAMethod(AFAMethod):
    """A dummy AFAMethod for testing purposes. Always chooses the next feature to observe in order."""

    def __init__(self, device: torch.device, n_classes: int):
        self._device = device
        self.n_classes = n_classes

    @override
    def select(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
        features: Features,
        label: Label,
    ) -> AFASelection:
        original_device = masked_features.device

        masked_features = masked_features.to(self._device)
        feature_mask = feature_mask.to(self._device)

        # Choose the next unobserved feature
        unobserved_features = (~feature_mask).nonzero(as_tuple=True)[1]
        if unobserved_features.numel() == 0:
            return torch.tensor(0, device=masked_features.device)
        selection = unobserved_features[0] + 1

        return selection.to(original_device)

    @override
    def predict(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
        features: Features,
        label: Label,
    ) -> Label:
        """Return a random prediction from the classes."""
        original_device = masked_features.device

        masked_features = masked_features.to(self._device)
        feature_mask = feature_mask.to(self._device)

        # Pick a random class from the classes
        prediction = torch.randint(
            0,
            self.n_classes,
            (masked_features.shape[0],),
            device=masked_features.device,
        )
        # One-hot encode the prediction
        prediction = torch.nn.functional.one_hot(
            prediction, num_classes=self.n_classes
        ).float()

        return prediction.to(original_device)

    @override
    def save(self, path: Path) -> None:
        """Save the method to a folder."""
        torch.save(
            {
                "n_classes": self.n_classes,
            },
            path / "method.pt",
        )

    @classmethod
    @override
    def load(cls, path: Path, device: torch.device) -> Self:
        """Load the method from a folder."""
        data = torch.load(path / "method.pt")
        return cls(device, data["n_classes"])

    @override
    def to(self, device: torch.device) -> Self:
        self._device = device
        return self

    @property
    @override
    def device(self) -> torch.device:
        return self._device


@final
class RandomClassificationAFAMethod(AFAMethod):
    """An AFAMethod that randomly selects new features but uses a trained classifier for predictions."""

    def __init__(
        self, afa_classifier: AFAClassifier, device: torch.device | None = None
    ):
        if device is None:
            self._device = afa_classifier.device
            self.afa_classifier = afa_classifier
        else:
            self._device = device
            self.afa_classifier = afa_classifier.to(device)

    @override
    def select(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
        features: Features,
        label: Label,
    ) -> AFASelection:
        """Chooses to observe a random feature from the masked features (or stop collecting features)."""
        original_device = masked_features.device

        masked_features = masked_features.to(self._device)
        feature_mask = feature_mask.to(self._device)

        # Sample from unobserved features uniformly
        probs = (~feature_mask).float()

        # Avoid division by zero
        row_sums = probs.sum(dim=1, keepdim=True)
        probs = torch.where(
            row_sums > 0, probs / row_sums, probs
        )  # normalize or leave zeros

        # Sample one index per row
        sampled = torch.multinomial(probs, num_samples=1)
        selection = sampled.squeeze(1)  # (B, 1) → (B,)

        return selection.to(original_device)

    @override
    def predict(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
        features: Features,
        label: Label,
    ) -> Label:
        """Return a prediction using the classifier."""
        original_device = masked_features.device

        return self.afa_classifier(
            masked_features.to(self._device), feature_mask.to(self._device)
        ).to(original_device)

    @override
    def save(self, path: Path) -> None:
        """Save the method to a folder."""
        self.afa_classifier.save(path / "classifier.pt")
        # Write the classifier class name to a file
        with open(path / "classifier_class_name.txt", "w") as f:
            f.write(self.afa_classifier.__class__.__name__)

    @classmethod
    @override
    def load(cls, path: Path, device: torch.device) -> Self:
        """Load the method from a file."""
        with open(path / "classifier_class_name.txt") as f:
            classifier_class_name = f.read()
        from common.registry import AFA_CLASSIFIER_REGISTRY

        afa_classifier = AFA_CLASSIFIER_REGISTRY[classifier_class_name].load(
            path / "classifier.pt", device
        )
        return cls(afa_classifier, device)

    @override
    def to(self, device: torch.device) -> Self:
        self._device = device
        self.afa_classifier.to(self._device)
        return self

    @property
    @override
    def device(self) -> torch.device:
        return self._device


@final
class SequentialClassificationAFAMethod(AFAMethod):
    """An AFAMethod that selects features sequentially and uses a trained classifier to predict labels."""

    def __init__(
        self, afa_classifier: AFAClassifier, device: torch.device | None = None
    ):
        if device is None:
            self._device = afa_classifier.device
            self.afa_classifier = afa_classifier
        else:
            self._device = device
            self.afa_classifier = afa_classifier.to(device)

    @override
    def select(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
        features: Features,
        label: Label,
    ) -> AFASelection:
        original_device = masked_features.device

        masked_features = masked_features.to(self._device)
        feature_mask = feature_mask.to(self._device)

        # Choose the next unobserved feature
        unobserved_features = (~feature_mask).nonzero(as_tuple=True)[1]
        if unobserved_features.numel() == 0:
            return torch.tensor(0, device=masked_features.device)
        selection = unobserved_features[0] + 1

        return selection.to(original_device)

    @override
    def predict(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
        features: Features,
        label: Label,
    ) -> Label:
        """Return a prediction using the classifier."""
        original_device = masked_features.device

        return self.afa_classifier(
            masked_features.to(self._device), feature_mask.to(self._device)
        ).to(original_device)

    @override
    def save(self, path: Path) -> None:
        """Save the method to a folder."""
        self.afa_classifier.save(path / "classifier.pt")
        # Write the classifier class name to a file
        with open(path / "classifier_class_name.txt", "w") as f:
            f.write(self.afa_classifier.__class__.__name__)

    @classmethod
    @override
    def load(cls, path: Path, device: torch.device) -> Self:
        """Load the method from a file."""
        with open(path / "classifier_class_name.txt") as f:
            classifier_class_name = f.read()
        from common.registry import AFA_CLASSIFIER_REGISTRY

        afa_classifier = AFA_CLASSIFIER_REGISTRY[classifier_class_name].load(
            path / "classifier.pt", device
        )
        return cls(afa_classifier, device)

    @override
    def to(self, device: torch.device) -> Self:
        self._device = device
        self.afa_classifier.to(self._device)
        return self

    @property
    @override
    def device(self) -> torch.device:
        return self._device


@final
class AFAContextSmartMethod(AFAMethod):
    """Always selects the first feature and then selects three other features depending on the context."""

    def __init__(self, device: torch.device):
        self._device = device
        self.n_classes = 8

    @override
    def select(
        self,
        masked_features: AFASelection,
        feature_mask: AFASelection,
        features: Features,
        label: Label,
    ) -> AFASelection:
        original_device = masked_features.device

        return afacontext_optimal_selection(
            masked_features=masked_features.to(self._device),
            feature_mask=feature_mask.to(self._device),
        ).to(original_device)

    @override
    def predict(
        self,
        masked_features: AFASelection,
        feature_mask: AFASelection,
        features: Features,
        label: Label,
    ) -> Label:
        # Guess class randomly
        return torch.randint(
            0,
            self.n_classes,
            (masked_features.shape[0],),
            device=masked_features.device,
        )

    @override
    def save(self, path: Path) -> None:
        torch.save({}, path)

    @classmethod
    @override
    def load(cls, path: Path, device: torch.device) -> Self:
        _ = torch.load(path)
        return cls(device)

    @override
    def to(self, device: torch.device) -> Self:
        self._device = device
        return self

    @property
    @override
    def device(self) -> torch.device:
        return self._device
