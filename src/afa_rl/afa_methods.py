from dataclasses import dataclass
from pathlib import Path
from typing import Self
import torch
from tensordict import TensorDict
from torchrl.envs import ExplorationType, set_exploration_type
from afa_rl.agents import Agent

from afa_rl.custom_types import NNMaskedClassifier
from afa_rl.utils import afacontext_optimal_selection
from common.custom_types import (
    AFAMethod,
    AFASelection,
    FeatureMask,
    Label,
    MaskedFeatures,
)


def get_td_from_masked_features(
    masked_features: MaskedFeatures,
    feature_mask: FeatureMask,
) -> TensorDict:
    """
    Creates a TensorDict including the keys
    - "action_mask"
    - "masked_features"
    - "feature_mask"
    from the masked features and the feature mask.
    """
    # The action mask is almost the same as the negated feature mask but with one extra element
    action_mask = torch.ones(
        masked_features.shape[0],
        masked_features.shape[1],
        dtype=torch.bool,
    )
    action_mask = ~feature_mask

    td = TensorDict(
        {
            "action_mask": action_mask,
            "masked_features": masked_features,
            "feature_mask": feature_mask,
        },
        batch_size=masked_features.shape[0],
    )

    return td


@dataclass
class RLAFAMethod(AFAMethod):
    """Implements the AFAMethod protocol for an Agent together with a classifier."""

    agent: Agent
    classifier: NNMaskedClassifier

    def __post_init__(self):
        # Set the agent to eval mode
        self.classifier.eval()

    def select(
        self, masked_features: MaskedFeatures, feature_mask: FeatureMask
    ) -> AFASelection:
        td = get_td_from_masked_features(masked_features, feature_mask)

        # Apply the agent's policy to the tensordict
        with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
            td = self.agent.policy(td)

        # Get the action from the tensordict
        afa_selection = td["action"].unsqueeze(-1)

        return afa_selection

    def predict(
        self, masked_features: MaskedFeatures, feature_mask: FeatureMask
    ) -> Label:
        with torch.no_grad():
            logits = self.classifier(masked_features, feature_mask)
        probs: Label = logits.softmax(dim=-1)
        return probs

    def save(self, path: Path):
        # Save the agent in a "agents" folder
        agent_path = path / "agent"
        self.agent.save(agent_path)

        # Save the classifier in a file named "classifier.pt"
        classifier_path = path / "classifier.pt"
        torch.save(self.classifier.cpu(), classifier_path)

    @classmethod
    def load(cls, path: Path, device: torch.device) -> Self:
        agent = Agent.load(path / "agent")
        agent.device = device

        classifier = torch.load(
            path / "classifier.pt", weights_only=False, map_location=device
        )
        classifier = classifier.to(device)

        return cls(
            agent=agent,
            classifier=classifier,
        )


class RandomDummyAFAMethod(AFAMethod):
    """
    A dummy AFAMethod for testing purposes. Chooses a random feature to observe from the masked features.
    """

    def __init__(self, device: torch.device, n_classes: int):
        self.device = device
        self.n_classes = n_classes

    def select(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
    ) -> AFASelection:
        """
        Chooses to observe a random feature from the masked features (or stop collecting features).
        """
        masked_features = masked_features.to(self.device)
        feature_mask = feature_mask.to(self.device)

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

        return selection

    def predict(
        self, masked_features: MaskedFeatures, feature_mask: FeatureMask
    ) -> Label:
        """
        Returns a random prediction from the classes.
        """
        masked_features = masked_features.to(self.device)
        feature_mask = feature_mask.to(self.device)

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

        return prediction

    def save(self, path: Path) -> None:
        """
        Saves the method to a file.
        """
        torch.save(
            {
                "n_classes": self.n_classes,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path, device: torch.device) -> Self:
        """
        Loads the method from a file.
        """
        data = torch.load(path)
        return cls(device, data["n_classes"])


class SequentialDummyAFAMethod(AFAMethod):
    """
    A dummy AFAMethod for testing purposes. Always chooses the next feature to observe in order.
    """

    def __init__(self, device: torch.device, n_classes: int):
        self.device = device
        self.n_classes = n_classes

    def select(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
    ) -> AFASelection:
        masked_features = masked_features.to(self.device)
        feature_mask = feature_mask.to(self.device)

        # Choose the next unobserved feature
        unobserved_features = (~feature_mask).nonzero(as_tuple=True)[1]
        if unobserved_features.numel() == 0:
            return torch.tensor(0, device=masked_features.device)
        selection = unobserved_features[0] + 1

        return selection

    def predict(
        self, masked_features: MaskedFeatures, feature_mask: FeatureMask
    ) -> Label:
        """
        Returns a random prediction from the classes.
        """
        masked_features = masked_features.to(self.device)
        feature_mask = feature_mask.to(self.device)

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

        return prediction

    def save(self, path: Path) -> None:
        """
        Saves the method to a file.
        """
        torch.save(
            {
                "n_classes": self.n_classes,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path, device: torch.device) -> Self:
        """
        Loads the method from a file.
        """
        data = torch.load(path)
        return cls(device, data["n_classes"])


class AFAContextSmartMethod(AFAMethod):
    """Always selects the first feature and then selects three other features depending on the context."""

    def __init__(self):
        self.n_classes = 8

    def select(self, masked_features: AFASelection, feature_mask: AFASelection) -> AFASelection:
        return afacontext_optimal_selection(
            masked_features=masked_features,
            feature_mask=feature_mask,
        )

    def predict(self, masked_features: AFASelection, feature_mask: AFASelection) -> Label:
        # Guess class randomly
        return torch.randint(
            0,
            self.n_classes,
            (masked_features.shape[0],),
            device=masked_features.device,
        )


    def save(self, path: Path) -> None:
        torch.save({
        }, path)

    @classmethod
    def load(cls, path: Path, device: torch.device) -> Self:
        data = torch.load(path)
        return cls()
