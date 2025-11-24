import torch
from jaxtyping import Bool
from torch import Tensor

from afabench.afa_rl.custom_types import (
    AFAReward,
    AFARewardFn,
)
from afabench.afa_rl.kachuee2019.models import Kachuee2019PQModule
from afabench.common.custom_types import (
    AFASelection,
    FeatureMask,
    Features,
    Label,
    MaskedFeatures,
)


def calc_reward(conf_a: Tensor, conf_b: Tensor, method: str):
    """
    Calculates the reward according to eq. (7) in "Opportunistic Learning: Budgeted Cost-Sensitive Learning from Data Streams"

    Args:
        conf_a (Tensor of shape (batch_size, n_classes)): confidence for feature vector without new feature acquired
        conf_b (Tensor of shape (batch_size, n_classes)): confidence for feature vector with new feature acquired

    """
    if method == "softmax":
        reward = torch.abs(conf_a.max(dim=-1)[0] - conf_b.max(dim=-1)[0])
    elif method == "Bayesian-L1":
        reward = torch.abs(conf_a - conf_b).sum(dim=-1)
    elif method == "Bayesian-L2":
        reward = ((conf_a - conf_b) ** 2.0).sum(dim=-1)
    else:
        raise NotImplementedError("Method is not supported:", method)
    return reward


def get_kachuee2019_reward_fn(
    pq_module: Kachuee2019PQModule,
    method: str,
    mcdrop_samples: int,
    acquisition_costs: torch.Tensor,
) -> AFARewardFn:
    """
    The reward function for kachuee2019.

    The agent receives a reward at each step of the episode, equal to the relative confidence change.

    Args:
        - `method` is one of {"softmax", "Bayesian-L1", "Bayesian-L2"}
        - `mcdrop_samples` determines how many samples to average over to get class probabilities

    """

    def f(
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
        new_masked_features: MaskedFeatures,
        new_feature_mask: FeatureMask,
        afa_selection: AFASelection,
        _features: Features,
        _label: Label,
        done: Bool[Tensor, "*batch 1"],
    ) -> AFAReward:
        # Acquisition cost per feature
        newly_acquired = (new_feature_mask & ~feature_mask).to(torch.float32)
        reward = -(newly_acquired * acquisition_costs).sum(dim=-1)
        reward = reward.squeeze(-1)

        conf_a = pq_module.confidence(
            masked_features, mcdrop_samples=mcdrop_samples
        )
        conf_b = pq_module.confidence(
            new_masked_features, mcdrop_samples=mcdrop_samples
        )
        reward = reward + calc_reward(conf_a, conf_b, method=method)
        return reward

    return f
