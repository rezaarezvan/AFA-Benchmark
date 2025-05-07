from dataclasses import dataclass
from torch import nn
from jaxtyping import Bool
from tensordict.nn import (
    ProbabilisticTensorDictSequential,
    TensorDictModule,
    TensorDictSequential,
)
from torch import Tensor
from torch.distributions import Categorical
from torchrl.modules import (
    ProbabilisticActor,
)
from torchrl_agents import serializable, unserializable
from torchrl_agents.ppo import PPOAgent

from afa_rl.zannone2019.models import PointNet
from common.custom_types import FeatureMask, MaskedFeatures


class Zannone2019ValueNet(nn.Module):
    def __init__(self, latent_size: int):
        super().__init__()
        self.latent_size = latent_size

        self.net = nn.Sequential(nn.Linear(latent_size, 1))

    def forward(self, mu: Tensor):
        return self.net(mu)


class Zannone2019PolicyNet(nn.Module):
    def __init__(self, latent_size: int, n_actions: int):
        super().__init__()
        self.latent_size = latent_size
        self.n_actions = n_actions

        self.net = nn.Sequential(nn.Linear(latent_size, n_actions))

    def forward(
        self,
        mu: Tensor,
        action_mask: Bool[Tensor, "batch n_actions"],
    ):
        action_logits = self.net(mu)
        # By setting the logits of invalid actions to -inf, we prevent them from being selected.
        action_logits[~action_mask] = float("-inf")
        return action_logits


class Zannone2019CommonNet(nn.Module):
    def __init__(self, pointnet: PointNet, encoder: nn.Module):
        super().__init__()
        self.pointnet = pointnet
        self.encoder = encoder

    def forward(self, masked_features: MaskedFeatures, feature_mask: FeatureMask):
        pointnet_output = self.pointnet(masked_features, feature_mask)
        encoding = self.encoder(pointnet_output)
        mu = encoding[:, : encoding.shape[1] // 2]
        return mu


@dataclass(kw_only=True, eq=False, order=False)
class Zannone2019Agent(PPOAgent):
    pointnet: PointNet = unserializable()
    encoder: nn.Module = unserializable()
    latent_size: int = serializable()

    def pre_init_hook(self):
        self.common_module = TensorDictModule(
            Zannone2019CommonNet(
                pointnet=self.pointnet,
                encoder=self.encoder,
            ),
            in_keys=["masked_features", "feature_mask"],
            out_keys=["mu"],
        )

    def get_policy_module(self) -> ProbabilisticTensorDictSequential:
        policy_head = Zannone2019PolicyNet(
            latent_size=self.latent_size,
            n_actions=self.action_spec.n,
        )
        policy_module = TensorDictSequential(
            [
                self.common_module,
                TensorDictModule(
                    policy_head,
                    in_keys=["mu", "action_mask"],
                    out_keys=["logits"],
                ),
            ]
        )

        probabilistic_policy_module = ProbabilisticActor(
            module=policy_module,
            spec=self.action_spec,
            in_keys=["logits"],
            distribution_class=Categorical,
            return_log_prob=True,
        )

        return probabilistic_policy_module

    def get_state_value_module(self) -> TensorDictModule:
        value_head = Zannone2019ValueNet(
            latent_size=self.latent_size,
        )
        state_value_module = TensorDictSequential(
            [
                self.common_module,
                TensorDictModule(
                    value_head,
                    in_keys=["mu"],
                    out_keys=["state_value"],
                ),
            ]
        )

        return state_value_module
