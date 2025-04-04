import torch
from tensordict import TensorDictBase
from tensordict.nn import TensorDictSequential
from torch import Tensor, nn, optim
from torchrl.data import (
    LazyTensorStorage,
    PrioritizedSampler,
    TensorDictReplayBuffer,
    TensorSpec,
)
from torchrl.modules import MLP, EGreedyModule, QValueActor
from torchrl.objectives import DQNLoss, SoftUpdate


class Shim2018ValueModule(nn.Module):
    def __init__(self, embedding_size, action_size, num_cells, device):
        super().__init__()
        self.embedding_size = embedding_size
        self.action_size = action_size
        self.num_cells = num_cells
        self.device = device

        self.net = MLP(
            in_features=embedding_size,
            out_features=action_size,
            num_cells=num_cells,
            device=device,
        )

    def forward(self, embedding, action_mask):
        qvalues = self.net(embedding)
        # By setting the Q-values of invalid actions to -inf, we prevent them from being selected greedily.
        qvalues[~action_mask] = float("-inf")
        return qvalues


class Shim2018Agent:
    def __init__(
        self,
        embedding_size: int,
        action_spec: TensorSpec,
        lr: float,
        update_tau: float,
        eps_init: float,
        eps_end: float,
        eps_steps: int,
        device: torch.device,
        replay_buffer_batch_size: int,  # batch size when sampling from replay buffer
        replay_buffer_size: int,  # how large the replay buffer is
        num_optim: int,  # number of times to sample from the replay buffer and train the agent
        replay_buffer_alpha: float,
        replay_buffer_beta: float,
    ):
        self.embedding_size = embedding_size
        self.action_spec = action_spec
        self.lr = lr
        self.update_tau = update_tau
        self.eps_init = eps_init
        self.eps_end = eps_end
        self.eps_steps = eps_steps
        self.device = device
        self.replay_buffer_batch_size = replay_buffer_batch_size
        self.replay_buffer_size = replay_buffer_size
        self.num_optim = num_optim
        self.replay_buffer_alpha = replay_buffer_alpha
        self.replay_buffer_beta = replay_buffer_beta

        self.value_module = Shim2018ValueModule(
            embedding_size=self.embedding_size,
            action_size=self.action_spec.n,
            num_cells=[32, 32],
            device=self.device,
        )
        self.value_network = QValueActor(
            module=self.value_module,
            in_keys=["embedding", "action_mask"],
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
            # double_dqn=False,
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

    # def save(self, filepath: str):
    #     checkpoint = {
    #         "embedding_size": self.embedding_size,
    #         "action_spec": self.action_spec,  # Ensure action_spec is serializable
    #         "lr": self.lr,
    #         "update_tau": self.update_tau,
    #         "eps": self.eps,
    #         "device": str(self.device),  # Convert device to string for compatibility
    #         "model_state_dict": self.value_module.state_dict(),
    #         "optimizer_state_dict": self.optim.state_dict(),
    #     }
    #     torch.save(checkpoint, filepath)

    # @staticmethod
    # def load(filepath: str, device: torch.device) -> "ShimQAgent":
    #     checkpoint = torch.load(filepath, map_location=device, weights_only=False)

    #     agent = ShimQAgent(
    #         embedding_size=checkpoint["embedding_size"],
    #         action_spec=checkpoint[
    #             "action_spec"
    #         ],  # Ensure action_spec is reconstructed properly
    #         lr=checkpoint["lr"],
    #         update_tau=checkpoint["update_tau"],
    #         eps=checkpoint["eps"],
    #         device=device,
    #     )

    #     # Load model and optimizer state
    #     agent.value_module.load_state_dict(checkpoint["model_state_dict"])
    #     agent.optim.load_state_dict(checkpoint["optimizer_state_dict"])

    #     return agent
