import torch
from jaxtyping import Bool
from torch import Tensor

from afabench.afa_rl.custom_types import AFAReward, AFARewardFn
from afabench.common.custom_types import (
    AFASelection,
    FeatureMask,
    Features,
    Label,
    MaskedFeatures,
    SelectionMask,
)


def get_fixed_reward_reward_fn(
    reward_for_stop: float, reward_otherwise: float
) -> AFARewardFn:
    # Return
    def f(
        masked_features: MaskedFeatures,  # current masked features  # noqa: ARG001
        feature_mask: FeatureMask,  # current feature mask  # noqa: ARG001
        selection_mask: SelectionMask,  # noqa: ARG001
        new_masked_features: MaskedFeatures,  # new masked features  # noqa: ARG001
        new_feature_mask: FeatureMask,  # new feature mask  # noqa: ARG001
        new_selection_mask: SelectionMask,  # noqa: ARG001
        selection: AFASelection,  # noqa: ARG001
        features: Features,  # noqa: ARG001
        label: Label,  # noqa: ARG001
        done: Bool[Tensor, "*batch 1"],  # done key
    ) -> AFAReward:
        reward = reward_otherwise * torch.ones_like(done, dtype=torch.float32)
        done_mask = done.squeeze(-1)
        reward[done_mask] = reward_for_stop
        # Ensure reward maintains the same shape as done
        return reward.view_as(done).to(torch.float32)

    return f
