from dataclasses import dataclass
from pathlib import Path
from typing import Self, final, override
import torch
from tensordict import TensorDict
from torchrl.envs import ExplorationType, set_exploration_type
from afa_rl.agents import Agent

from common.custom_types import (
    AFAClassifier,
    AFAMethod,
    AFASelection,
    FeatureMask,
    Label,
    MaskedFeatures,
)
from common.registry import get_afa_classifier_class

# Agent (and therefore RLAFAMethod) needs all subclasses of Agent to be imported
from afa_rl.shim2018.agents import Shim2018Agent


def get_td_from_masked_features(
    masked_features: MaskedFeatures,
    feature_mask: FeatureMask,
) -> TensorDict:
    """Create a TensorDict suitable as input to AFA RL agents.

    The keys are:
    - "action_mask"
    - "masked_features"
    - "feature_mask"
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
@final
class RLAFAMethod(AFAMethod):
    """Implements the AFAMethod protocol for an Agent together with a classifier."""

    agent: Agent
    afa_classifier: AFAClassifier
    _device: torch.device = torch.device("cpu")

    def __post_init__(self):
        # Move agent and classifier to the specified device
        self.agent = self.agent.to(self._device)
        self.afa_classifier = self.afa_classifier.to(self._device)

    @override
    def select(
        self, masked_features: MaskedFeatures, feature_mask: FeatureMask
    ) -> AFASelection:
        original_device = masked_features.device

        masked_features = masked_features.to(self._device)
        feature_mask = feature_mask.to(self._device)

        td = get_td_from_masked_features(masked_features, feature_mask)

        # Apply the agent's policy to the tensordict
        with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
            td = self.agent.policy(td)

        # Get the action from the tensordict
        afa_selection = td["action"].unsqueeze(-1)

        return afa_selection.to(original_device)

    @override
    def predict(
        self, masked_features: MaskedFeatures, feature_mask: FeatureMask
    ) -> Label:
        original_device = masked_features.device

        masked_features = masked_features.to(self._device)
        feature_mask = feature_mask.to(self._device)

        with torch.no_grad():
            probs = self.afa_classifier(masked_features, feature_mask)
        return probs.to(original_device)

    @override
    def save(self, path: Path):
        self.agent.save(path / "agent")
        self.afa_classifier.save(path / "classifier.pt")
        with open(path / "classifier_class_name.txt", "w") as f:
            f.write(self.afa_classifier.__class__.__name__)

    @classmethod
    @override
    def load(cls, path: Path, device: torch.device) -> Self:
        agent = Agent.load(path / "agent")
        agent = agent.to(device)

        with open(path / "classifier_class_name.txt") as f:
            classifier_class_name = f.read()
        afa_classifier = get_afa_classifier_class(classifier_class_name).load(
            path / "classifier.pt", device=device
        )

        return cls(agent=agent, afa_classifier=afa_classifier, _device=device)

    @override
    def to(self, device: torch.device) -> Self:
        self.agent = self.agent.to(device)
        self.afa_classifier = self.afa_classifier.to(device)
        return self

    @property
    @override
    def device(self) -> torch.device:
        return self._device
