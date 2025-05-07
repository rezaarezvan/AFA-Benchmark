from dataclasses import dataclass
from pathlib import Path
from typing import Self
import torch
from tensordict import TensorDict
from torchrl.envs import ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor
from afa_rl.agents import Agent

from afa_rl.afa_methods import get_td_from_masked_features
from afa_rl.custom_types import NNMaskedClassifier
from afa_rl.zannone2019.models import Zannone2019PretrainingModel
from common.custom_types import AFAMethod, AFASelection, FeatureMask, Label, MaskedFeatures

class Zannone2019AFAMethod(AFAMethod):
    """
    Implements the AFAMethod protocol for the Zannone2019 agent.
    """

    def __init__(
        self,
        device: torch.device,
        probabilistic_policy_module: ProbabilisticActor,
        pretrained_model: Zannone2019PretrainingModel
    ):
        self.device = device
        self.probabilistic_policy_module = probabilistic_policy_module
        self.pretrained_model = pretrained_model

    def select(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
    ) -> AFASelection:
        masked_features = masked_features.to(self.device)
        feature_mask = feature_mask.to(self.device)

        td = get_td_from_masked_features(masked_features, feature_mask)

        # Apply the agent's policy to the tensordict
        with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
            td = self.probabilistic_policy_module(td)

        # Get the action from the tensordict
        afa_selection = td["action"].unsqueeze(-1)

        return afa_selection

    def predict(self, masked_features: MaskedFeatures, feature_mask: FeatureMask) -> Label:
        masked_features = masked_features.to(self.device)
        feature_mask = feature_mask.to(self.device)

        with torch.no_grad():
            encoding, mu, logvar, z = self.pretrained_model.partial_vae.encode(masked_features, feature_mask)
            logits = self.pretrained_model.classifier(mu)

        probs: Label = logits.softmax(dim=-1)
        return probs

    def save(self, path: Path):
        torch.save(
            {
                "probabilistic_policy_module": self.probabilistic_policy_module.cpu(),
                "pretrained_model": self.pretrained_model.cpu(),
            },
            path,
        )

    @classmethod
    def load(cls, path: Path, device: torch.device) -> Self:
        """
        Loads the Zannone2019AFAMethod object, including its components.
        """
        data = torch.load(path, weights_only=False, map_location=device)

        probabilistic_policy_module = data["probabilistic_policy_module"].to(device)
        pretrained_model = data["pretrained_model"].to(device)

        return cls(
            device=device,
            probabilistic_policy_module=probabilistic_policy_module,
            pretrained_model=pretrained_model,
        )
