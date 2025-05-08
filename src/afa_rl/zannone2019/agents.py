
from torch import nn
import torch
from jaxtyping import Bool
from tensordict import TensorDictBase
from tensordict.nn import TensorDictModule, TensorDictSequential
from torch import Tensor, nn, optim
from torch.distributions import Categorical
from torchrl.data import (
    LazyTensorStorage,
    PrioritizedSampler,
    TensorDictReplayBuffer,
    TensorSpec,
)
from torchrl.modules import (
    MLP,
    ActorCriticOperator,
    ActorValueOperator,
    EGreedyModule,
    ProbabilisticActor,
    QValueActor,
    ValueOperator,
)
from torchrl.objectives import ClipPPOLoss, DQNLoss, SoftUpdate
from torchrl.objectives.value import GAE
from torchrl_agents.dqn import DQNAgent

from afa_rl.zannone2019.models import PointNet
from afa_rl.shim2018.models import Shim2018Embedder
from common.custom_types import FeatureMask, MaskedFeatures


class Zannone2019ValueModule(nn.Module):
    def __init__(
        self, latent_size: int
    ):
        super().__init__()
        self.latent_size = latent_size

        self.net = nn.Sequential(nn.Linear(latent_size, 1))

    def forward(self, mu: Tensor):
        return self.net(mu)


class Zannone2019PolicyModule(nn.Module):
    def __init__(
        self, latent_size: int, n_actions: int
    ):
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

class Zannone2019CommonModule(nn.Module):
    def __init__(
        self, pointnet: PointNet, encoder: nn.Module
    ):
        super().__init__()
        self.pointnet = pointnet
        self.encoder = encoder

    def forward(self, masked_features: MaskedFeatures, feature_mask: FeatureMask):
        pointnet_output = self.pointnet(masked_features, feature_mask)
        encoding = self.encoder(pointnet_output)
        mu = encoding[:, : encoding.shape[1] // 2]
        return mu



class Zannone2019Agent:
    def __init__(
        self,
        device: torch.device,
        pointnet: PointNet,
        encoder: nn.Module,
        lr: float,
        latent_size: int,
        action_spec: TensorSpec,
        lmbda: float,
        clip_epsilon: float,
        entropy_bonus: bool,
        entropy_coef: float,
        max_grad_norm: float,
    ):
        self.device = device

        self.pointnet = pointnet.to(self.device)
        self.encoder = encoder.to(self.device)

        self.lr = lr
        self.latent_size = latent_size
        self.action_spec = action_spec
        self.lmbda = lmbda
        self.clip_epsilon = clip_epsilon
        self.entropy_bonus = entropy_bonus
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

        self.common_backbone = Zannone2019CommonModule(
            pointnet=self.pointnet,
            encoder=self.encoder,
        )

        self.policy_head = Zannone2019PolicyModule(
            latent_size=self.latent_size,
            n_actions=self.action_spec.n,
        )
        self.value_head = Zannone2019ValueModule(
            latent_size=self.latent_size,
        )

        actor_value_operator = ActorValueOperator(
            TensorDictModule(
                self.common_backbone,
                in_keys=["masked_features", "feature_mask"],
                out_keys=["mu"],
            ),
            TensorDictModule(
                self.policy_head,
                in_keys=["mu", "action_mask"],
                out_keys=["logits"]
            ),
            TensorDictModule(
                self.value_head,
                in_keys=["mu"],
                out_keys=["state_value"]
            ),
        ).to(self.device)

        self.policy_module = actor_value_operator.get_policy_operator()
        self.value_module = actor_value_operator.get_value_operator()

        self.probabilistic_policy_module = ProbabilisticActor(
            module=self.policy_module,
            spec=self.action_spec,
            in_keys=["logits"],
            distribution_class=Categorical,
            return_log_prob=True,
        )

        self.loss_module = ClipPPOLoss(
            actor_network=self.probabilistic_policy_module,
            critic_network=self.value_module,
            clip_epsilon=self.clip_epsilon,
            entropy_bonus=self.entropy_bonus,
            entropy_coef=self.entropy_coef,
        )
        self.loss_keys = ["loss_critic", "loss_objective"]
        if self.entropy_bonus:
            self.loss_keys.append("loss_entropy")
        self.optim = optim.Adam(self.loss_module.parameters(), lr=self.lr)

        self.advantage_module = GAE(
            gamma=1, lmbda=self.lmbda, value_network=self.value_module, average_gae=True
        )

    def policy(self, td: TensorDictBase):
        td = self.probabilistic_policy_module(td)
        return td

    def process_batch(self, td):
        self.optim.zero_grad()

        self.advantage_module(td)

        loss = self.loss_module(td)
        loss_value = sum(loss[loss_key] for loss_key in self.loss_keys)
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(
            self.loss_module.parameters(), max_norm=self.max_grad_norm, norm_type=2
        )
        loss_value.backward()

        self.optim.step()
        # self.updater.step()

        return loss_value.mean()
