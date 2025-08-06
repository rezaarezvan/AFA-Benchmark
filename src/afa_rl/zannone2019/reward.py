from jaxtyping import Bool

import torch
import torch.nn.functional as F
from torch import Tensor

from afa_rl.custom_types import (
    AFAReward,
    AFARewardFn,
)
from afa_rl.zannone2019.models import (
    Zannone2019PretrainingModel,
)
from common.custom_types import (
    AFASelection,
    FeatureMask,
    Features,
    Label,
    MaskedFeatures,
)


def get_zannone2019_reward_fn(
    pretrained_model: Zannone2019PretrainingModel, weights: Tensor
) -> AFARewardFn:
    """The reward function for zannone2019.

    The agent receives a reward at each step of the episode, equal to the negative classification loss.
    """

    def f(
        _masked_features: MaskedFeatures,
        _feature_mask: FeatureMask,
        new_masked_features: MaskedFeatures,
        new_feature_mask: FeatureMask,
        _afa_selection: AFASelection,
        _features: Features,
        label: Label,
        _done: Bool[Tensor, "*batch 1"],
    ) -> AFAReward:
        # We don't get to observe the label
        new_augmented_masked_features = torch.cat(
            [new_masked_features, torch.zeros_like(label)], dim=-1
        )
        new_augmented_feature_mask = torch.cat(
            [new_feature_mask, torch.full_like(label, False)], dim=-1
        )
        _encoding, mu, _logvar, z = pretrained_model.partial_vae.encode(
            new_augmented_masked_features, new_augmented_feature_mask
        )
        logits = pretrained_model.classifier(mu)
        reward = -F.cross_entropy(logits, label, weight=weights, reduction="none")

        return reward

    return f
