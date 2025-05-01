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

from afa_rl.models import PointNet, ShimEmbedder
from common.custom_types import FeatureMask, MaskedFeatures


class Shim2018ValueModule(nn.Module):
    def __init__(
        self, embedder: ShimEmbedder, embedding_size, action_size, num_cells
    ):
        super().__init__()
        self.embedder = embedder
        self.embedding_size = embedding_size
        self.action_size = action_size
        self.num_cells = num_cells

        self.net = MLP(
            in_features=embedding_size,
            out_features=action_size,
            num_cells=num_cells,
        )

    def forward(
        self, masked_features: MaskedFeatures, feature_mask: FeatureMask, action_mask
    ):
        # We do not want to update the embedder weights using the Q-values, this is done separately in the training loop
        with torch.no_grad():
            embedding = self.embedder(masked_features, feature_mask)
        qvalues = self.net(embedding)
        # By setting the Q-values of invalid actions to -inf, we prevent them from being selected greedily.
        qvalues[~action_mask] = float("-inf")
        return qvalues


class Shim2018Agent:
    def __init__(
        self,
        device: torch.device,
        embedder: ShimEmbedder,
        embedding_size: int,
        action_spec: TensorSpec,
        lr: float,
        update_tau: float,
        eps_init: float,
        eps_end: float,
        eps_steps: int,
        replay_buffer_batch_size: int,  # batch size when sampling from replay buffer
        replay_buffer_size: int,  # how large the replay buffer is
        num_optim: int,  # number of times to sample from the replay buffer and train the agent
        replay_buffer_alpha: float,
        replay_buffer_beta: float,
        max_grad_norm: float,
    ):
        self.device = device
        self.embedder = embedder.to(self.device)
        self.embedding_size = embedding_size
        self.action_spec = action_spec
        self.lr = lr
        self.update_tau = update_tau
        self.eps_init = eps_init
        self.eps_end = eps_end
        self.eps_steps = eps_steps
        self.replay_buffer_batch_size = replay_buffer_batch_size
        self.replay_buffer_size = replay_buffer_size
        self.num_optim = num_optim
        self.replay_buffer_alpha = replay_buffer_alpha
        self.replay_buffer_beta = replay_buffer_beta
        self.max_grad_norm = max_grad_norm

        self.value_module = Shim2018ValueModule(
            embedder=self.embedder,
            embedding_size=self.embedding_size,
            action_size=self.action_spec.n,
            num_cells=[32, 32],
        ).to(self.device)
        self.value_network = QValueActor(
            module=self.value_module,
            in_keys=["masked_features", "feature_mask", "action_mask"],
            spec=self.action_spec,
        )
        self.egreedy_module = EGreedyModule(
            spec=self.action_spec,
            eps_init=self.eps_init,
            eps_end=self.eps_end,
            annealing_num_steps=self.eps_steps,
            action_mask_key="action_mask",  # prevents agent from choosing invalid actions non-greedily
        )
        self.egreedy_actor = TensorDictSequential(
            [self.value_network, self.egreedy_module]
        )
        self.loss_module = DQNLoss(
            value_network=self.value_network,
            action_space=self.action_spec,
            double_dqn=True,
            delay_value=True,
            loss_function="l2",
        )
        self.loss_module.make_value_estimator(gamma=1)
        self.optim = optim.Adam(self.loss_module.parameters(), lr=self.lr)
        self.updater = SoftUpdate(self.loss_module, tau=self.update_tau)

        # Replay buffer stored on device
        self.replay_buffer = TensorDictReplayBuffer(
            storage=LazyTensorStorage(
                max_size=self.replay_buffer_size, device=self.device
            ),
            batch_size=self.replay_buffer_batch_size,
            sampler=PrioritizedSampler(
                max_capacity=self.replay_buffer_size,
                alpha=self.replay_buffer_alpha,
                beta=self.replay_buffer_beta,
            ),
        )

    def policy(self, td: TensorDictBase):
        td = self.egreedy_actor(td)
        return td

    def greedy_policy(self, td: TensorDictBase):
        """
        Greedily select actions based on the Q-values of the value network.
        """
        td = self.value_network(td)
        return td

    def add_to_replay_buffer(self, td: TensorDictBase):
        """
        Adds a batch of data to replay buffer
        """
        self.replay_buffer.extend(td)

    def _train_sample(self) -> Tensor:
        """
        Sample a batch from the replay buffer and train the agent, returning losses
        """
        # if len(self.replay_buffer) < self.replay_buffer_batch_size:
        #     return

        td = self.replay_buffer.sample()
        self.optim.zero_grad()
        loss = self.loss_module(td)["loss"]
        loss.backward()
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(
            self.loss_module.parameters(), max_norm=self.max_grad_norm, norm_type=2
        )
        self.optim.step()
        self.updater.step()

        # Update the priorities of the sampled batch
        self.replay_buffer.update_tensordict_priority(td)

        return loss

    def train(self):
        """
        Train the agent by sampling from the replay buffer several times, returning average loss
        """

        loss = torch.zeros(
            self.replay_buffer_batch_size, dtype=torch.float32, device=self.device
        )

        for _ in range(self.num_optim):
            loss += self._train_sample()

        return loss.mean().item() / self.num_optim


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
