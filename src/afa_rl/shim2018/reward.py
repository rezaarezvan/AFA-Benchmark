import torch
from jaxtyping import Bool
from torch import Tensor
from torch.nn import functional as F

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
    pretrained_model: LitShim2018EmbedderClassifier,
    weights: Tensor,
    acquisition_cost: float,
) -> AFARewardFn:
    """
    Return the reward function for shim2018.

    The agent receives the negative classification loss as reward at the end of the episode, and also a fixed
    negative reward for each feature selected, encouraging it to select fewer features.
    """

    def f(
        masked_features: MaskedFeatures,  # noqa: ARG001
        feature_mask: FeatureMask,  # noqa: ARG001
        new_masked_features: MaskedFeatures,
        new_feature_mask: FeatureMask,
        afa_selection: AFASelection,
        features: Features,  # noqa: ARG001
        label: Label,
        done: Bool[Tensor, "*batch 1"],
    ) -> AFAReward:
        # Always a constant negative reward to encourage fewer feature selections
        reward = -acquisition_cost * torch.ones_like(
            afa_selection, dtype=torch.float32
        )

        done_mask = done.squeeze(-1)

        if done_mask.any():
            _, logits = pretrained_model(
                new_masked_features[done_mask], new_feature_mask[done_mask]
            )
            reward[done_mask] = -F.cross_entropy(
                logits, label[done_mask], weight=weights
            )

        return reward

    return f
