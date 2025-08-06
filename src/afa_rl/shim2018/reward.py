from jaxtyping import Bool
from torch.nn import functional as F

import torch
from torch import Tensor

from afa_rl.custom_types import (
    AFAReward,
    AFARewardFn,
)
from afa_rl.shim2018.models import LitShim2018EmbedderClassifier
from common.custom_types import (
    AFASelection,
    FeatureMask,
    Features,
    Label,
    MaskedFeatures,
)


def get_shim2018_reward_fn(
    pretrained_model: LitShim2018EmbedderClassifier, weights: Tensor
) -> AFARewardFn:
    """The reward function for shim2018.

    The agent only receives a reward at the end of the episode, equal to the negative classification loss.
    """

    def f(
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
        new_masked_features: MaskedFeatures,
        new_feature_mask: FeatureMask,
        afa_selection: AFASelection,
        features: Features,
        label: Label,
        done: Bool[Tensor, "*batch 1"],
    ) -> AFAReward:
        reward = torch.zeros_like(afa_selection, dtype=torch.float32)

        done_mask = done.squeeze(-1)

        # FIX:
        if done_mask.any():
            _, logits = pretrained_model(
                new_masked_features[done_mask], new_feature_mask[done_mask]
            )
            reward[done_mask] = -F.cross_entropy(
                logits, label[done_mask], weight=weights
            )

        return reward

    return f
