import torch
from tensordict import TensorDictBase
from tensordict.nn import TensorDictSequential
from torch import nn, optim
from torchrl.data import TensorSpec
from torchrl.modules import MLP, EGreedyModule, QValueActor
from torchrl.objectives import DQNLoss, SoftUpdate

from afa_rl.utils import resample_invalid_actions


class ShimQAgent:
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
    ):
        self.embedding_size = embedding_size
        self.action_spec = action_spec
        self.lr = lr
        self.update_tau = update_tau
        self.eps_init = eps_init
        self.eps_end = eps_end
        self.eps_steps = eps_steps
        self.device = device

        # TODO: check if this action mask hack is problematic
        class ValueModule(nn.Module):
            def __init__(self, embedding_size, action_size, num_cells, device):
                super().__init__()
                self.net = MLP(
                    in_features=embedding_size,
                    out_features=action_size,
                    num_cells=num_cells,
                    device=device,
                )

            def forward(self, embedding):
                qvalues = self.net(embedding)
                # Actions that are not allowed are set equal to the smallest value
                # in the qvalues tensor. This is a hack to make sure that the
                # EGreedyModule will not choose these actions.
                # qvalues[~action_mask] = qvalues.min()
                # qvalues[~action_mask] = float("-inf")
                return qvalues

        self.value_module = ValueModule(
            embedding_size=self.embedding_size,
            action_size=self.action_spec.n,
            num_cells=[32, 32],
            device=self.device,
        )
        self.value_network = QValueActor(
            module=self.value_module,
            in_keys=["embedding"],
            spec=self.action_spec,
        )
        self.egreedy_module = EGreedyModule(
            spec=self.action_spec,
            eps_init=self.eps_init,
            eps_end=self.eps_end,
            annealing_num_steps=self.eps_steps,
            # It would be preferrable to use EGreedyModule's built-in action_mask_key but this
            # is ignored in deterministic modes unfortunately.
            # action_mask_key="action_mask",
        )
        self.egreedy_actor = TensorDictSequential(
            [self.value_network, self.egreedy_module]
        )
        self.loss_module = DQNLoss(
            value_network=self.value_network,
            action_space=self.action_spec,
            # double_dqn=True,
            double_dqn=False,
            delay_value=True,
            loss_function="l2",
        )
        self.loss_module.make_value_estimator(gamma=1)
        self.optim = optim.Adam(self.loss_module.parameters(), lr=self.lr)
        self.updater = SoftUpdate(self.loss_module, tau=self.update_tau)

    def policy(self, td: TensorDictBase):
        td = self.egreedy_actor(td)
        # EGreedyModule will still choose non-available actions (masked) when eps case is not triggered.
        # To prevent this, we choose a random action in case an invalid action is chosen.
        # td["action"] = resample_invalid_actions(td["action"], td["action_mask"], td["action_value"])
        return td

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
