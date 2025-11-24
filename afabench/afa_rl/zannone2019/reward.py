import torch
from jaxtyping import Bool
from torch import Tensor

from afabench.afa_rl.custom_types import (
    AFAReward,
    AFARewardFn,
)
from afabench.afa_rl.zannone2019.models import Zannone2019PretrainingModel
from afabench.common.custom_types import (
    AFASelection,
    FeatureMask,
    Features,
    Label,
    MaskedFeatures,
)


def get_zannone2019_reward_fn(
    pretrained_model: Zannone2019PretrainingModel,
    weights: Tensor,
    acquisition_costs: torch.Tensor,
) -> AFARewardFn:
    """The reward function for zannone2019."""

    def f(
        _masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
        new_masked_features: MaskedFeatures,
        new_feature_mask: FeatureMask,
        _afa_selection: AFASelection,
        _features: Features,
        label: Label,
        done: Bool[Tensor, "*batch 1"],
    ) -> AFAReward:
        # Acquisition cost per feature
        newly_acquired = (new_feature_mask & ~feature_mask).to(torch.float32)
        reward = -(newly_acquired * acquisition_costs).sum(dim=-1)
        reward = reward.squeeze(-1)

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
        predictions = logits.argmax(dim=-1, keepdim=True)
        integer_label = label.argmax(dim=-1, keepdim=True)
        reward = reward + (predictions == integer_label).to(
            torch.float32
        ).squeeze(-1)

        return reward

    return f
