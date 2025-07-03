from jaxtyping import Bool

import torch
from torch import Tensor

from afa_rl.custom_types import (
    AFAReward,
    AFARewardFn,
)
from common.custom_types import (
    AFAPredictFn,
    AFASelection,
    FeatureMask,
    Features,
    Label,
    MaskedFeatures,
)
from common.utils import eval_mode


def calc_reward(conf_a: Tensor, conf_b: Tensor, method: str):
    if method == "softmax":
        reward = torch.abs(conf_a.max() - conf_b.max())
    elif method == "Bayesian-L1":
        reward = torch.abs(conf_a - conf_b).sum()
    elif method == "Bayesian-L2":
        reward = ((conf_a - conf_b) ** 2.0).sum()
    else:
        raise NotImplementedError("Method is not supported:", method)
    return reward


def get_kachuee2019_reward_fn(afa_predict_fn: AFAPredictFn, method: str) -> AFARewardFn:
    """The reward function for kachuee2019.

    The agent receives a reward at each step of the episode, equal to the relative confidence change.

    method is one of {"softmax", "Bayesian-L1", "Bayesian-L2"}
    """

    def f(
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
        new_masked_features: MaskedFeatures,
        new_feature_mask: FeatureMask,
        _afa_selection: AFASelection,
        _features: Features,
        _label: Label,
        _done: Bool[Tensor, "*batch 1"],
    ) -> AFAReward:
        conf_a = afa_predict_fn(masked_features, feature_mask)
        conf_b = afa_predict_fn(new_masked_features, new_feature_mask)
        reward = calc_reward(conf_a, conf_b, method=method)
        return reward

    return f
