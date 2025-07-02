from jaxtyping import Bool

from torch import Tensor

from afa_rl.custom_types import (
    AFAReward,
    AFARewardFn,
)
from afa_rl.utils import weighted_cross_entropy
from afa_rl.zannone2019.models import Zannone2019AFAPredictFn
from common.custom_types import (
    AFASelection,
    FeatureMask,
    Features,
    Label,
    MaskedFeatures,
)


def get_zannone2019_reward_fn(
    afa_predict_fn: Zannone2019AFAPredictFn, weights: Tensor
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
        probs = afa_predict_fn(new_masked_features, new_feature_mask)
        reward = -weighted_cross_entropy(
            input_probs=probs, target_probs=label, weights=weights
        )

        return reward

    return f
