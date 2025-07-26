from pathlib import Path
from typing import Any, Callable, Self, cast, final, override
from omegaconf import OmegaConf
from tensordict import TensorDictBase
from torch import nn, optim
from jaxtyping import Bool
from tensordict.nn import (
    TensorDictModule,
    TensorDictSequential,
)
from torch import Tensor
import torch
from torch.distributions import Categorical
from torchrl.data import (
    LazyTensorStorage,
    SamplerWithoutReplacement,
    TensorDictReplayBuffer,
    TensorSpec,
)
from torchrl.envs import ExplorationType, set_exploration_type
from torchrl.modules import (
    MLP,
    ProbabilisticActor,
)
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from afa_rl.agents import Agent

from afa_rl.utils import module_norm
from afa_rl.zannone2019.models import PointNet
from common.config_classes import Zannone2019AgentConfig
from common.custom_types import FeatureMask, MaskedFeatures


@final
class Zannone2019ValueModule(nn.Module):
    def __init__(self, latent_size: int, num_cells: tuple[int, ...], dropout: float):
        super().__init__()
        self.latent_size = latent_size
        self.num_cells = num_cells
        self.dropout = dropout

        self.net = MLP(
            in_features=latent_size,
            out_features=1,
            num_cells=self.num_cells,
            dropout=self.dropout,
        )

    @override
    def forward(self, mu: Tensor):
        return self.net(mu)


@final
class Zannone2019PolicyModule(nn.Module):
    def __init__(
        self,
        latent_size: int,
        n_actions: int,
        num_cells: tuple[int, ...],
        dropout: float,
    ):
        super().__init__()
        self.latent_size = latent_size
        self.n_actions = n_actions
        self.num_cells = num_cells
        self.dropout = dropout

        self.net = MLP(
            in_features=latent_size,
            out_features=n_actions,
            num_cells=self.num_cells,
            dropout=self.dropout,
        )

    @override
    def forward(
        self,
        mu: Tensor,
        action_mask: Bool[Tensor, "batch n_actions"],
    ):
        action_logits = self.net(mu)
        # By setting the logits of invalid actions to -inf, we prevent them from being selected.
        action_logits[~action_mask] = float("-inf")
        return action_logits


@final
class Zannone2019CommonModule(nn.Module):
    def __init__(self, pointnet: PointNet, encoder: nn.Module):
        super().__init__()
        self.pointnet = pointnet
        self.encoder = encoder

    @override
    def forward(self, masked_features: MaskedFeatures, feature_mask: FeatureMask):
        pointnet_output = self.pointnet(masked_features, feature_mask)
        encoding = self.encoder(pointnet_output)
        mu = encoding[:, : encoding.shape[1] // 2]
        return mu


@final
class Zannone2019DummyCommonModule(nn.Module):
    def __init__(self):
        super().__init__()

    @override
    def forward(self, masked_features: MaskedFeatures, feature_mask: FeatureMask):
        return torch.cat([masked_features, feature_mask], dim=-1)


@final
class Zannone2019Agent(Agent):
    def __init__(
        self,
        cfg: Zannone2019AgentConfig,
        pointnet: PointNet,
        encoder: nn.Module,
        action_spec: TensorSpec,
        latent_size: int,
        action_mask_key: str,
        batch_size: int,
        module_device: torch.device,
        replay_buffer_device: torch.device,
    ):
        self.cfg = cfg
        self.pointnet = pointnet
        self.encoder = encoder
        self.action_spec = action_spec
        self.latent_size = latent_size
        self.action_mask_key = action_mask_key
        self.batch_size = batch_size
        self.module_device = module_device
        self.replay_buffer_device = replay_buffer_device

        self.common_module = Zannone2019CommonModule(
            pointnet=self.pointnet,
            encoder=self.encoder,
        ).to(self.module_device)
        # self.common_module = Zannone2019DummyCommonModule().to(self.module_device)
        self.common_tdmodule = TensorDictModule(
            module=self.common_module,
            in_keys=["masked_features", "feature_mask"],
            out_keys=["mu"],
        )
        self.policy_head = Zannone2019PolicyModule(
            latent_size=self.latent_size,
            # latent_size=20,
            n_actions=self.action_spec.n,  # pyright: ignore
            num_cells=tuple(self.cfg.policy_num_cells),
            dropout=self.cfg.policy_dropout,
        ).to(self.module_device)
        self.policy_tdmodule = TensorDictSequential(
            [
                self.common_tdmodule,
                TensorDictModule(
                    self.policy_head,
                    in_keys=["mu", self.action_mask_key],
                    out_keys=["logits"],
                ),
            ]
        )

        self.probabilistic_policy_tdmodule = ProbabilisticActor(
            module=self.policy_tdmodule,
            spec=self.action_spec,
            in_keys=["logits"],
            distribution_class=Categorical,
            return_log_prob=True,
        )

        self.value_head = Zannone2019ValueModule(
            latent_size=self.latent_size,
            # latent_size=20,
            num_cells=tuple(self.cfg.value_num_cells),
            dropout=self.cfg.value_dropout,
        ).to(self.module_device)

        self.state_value_tdmodule = TensorDictSequential(
            [
                self.common_tdmodule,
                TensorDictModule(
                    self.value_head,
                    in_keys=["mu"],
                    out_keys=["state_value"],
                ),
            ]
        )

        # self.advantage_module = GAE(
        #     gamma=self.cfg.gamma,
        #     lmbda=self.cfg.lmbda,
        #     value_network=self.state_value_tdmodule,
        #     average_gae=self.cfg.replay_buffer_batch_size
        #     > 1,  # we cannot average or calculate std with a single sample
        # )
        self.loss_tdmodule = ClipPPOLoss(
            actor_network=self.probabilistic_policy_tdmodule,
            critic_network=self.state_value_tdmodule,
            clip_epsilon=self.cfg.clip_epsilon,
            entropy_bonus=self.cfg.entropy_bonus,
            entropy_coef=self.cfg.entropy_coef,
            critic_coef=self.cfg.critic_coef,
            loss_critic_type=self.cfg.loss_critic_type,
        ).to(self.module_device)
        self.loss_keys = ["loss_objective", "loss_critic"] + (
            ["loss_entropy"] if self.cfg.entropy_bonus else []
        )
        self.optimizer = optim.Adam(self.loss_tdmodule.parameters(), lr=self.cfg.lr)
        self.replay_buffer = TensorDictReplayBuffer(
            storage=LazyTensorStorage(
                max_size=self.batch_size,
                device=torch.device(self.replay_buffer_device),
            ),
            sampler=SamplerWithoutReplacement(),
            batch_size=self.cfg.replay_buffer_batch_size,
        )

    @override
    def get_exploitative_policy(self) -> ProbabilisticActor:
        # No distinction between "exploitative" and "exploratory" modules
        # User has to set ExplorationType
        return self.probabilistic_policy_tdmodule

    @override
    def get_exploratory_policy(self) -> ProbabilisticActor:
        # No distinction between "exploitative" and "exploratory" modules
        # User has to set ExplorationType
        return self.probabilistic_policy_tdmodule

    @override
    def get_policy(self) -> ProbabilisticActor:
        return self.probabilistic_policy_tdmodule

    @override
    def process_batch(self, td: TensorDictBase) -> dict[str, Any]:
        assert td.batch_size == torch.Size((self.batch_size,)), "Batch size mismatch"

        # Initialize total loss dictionary
        total_loss_dict = {k: 0.0 for k in self.loss_keys}

        # Perform multiple epochs of training
        for _ in range(self.cfg.num_epochs):
            # Compute advantages each epoch
            # self.advantage_module(td)

            # Reset replay buffer each epoch
            self.replay_buffer.extend(td)

            for _ in range(self.batch_size // self.cfg.replay_buffer_batch_size):
                sampled_td = self.replay_buffer.sample()
                sampled_td = sampled_td.to(self.module_device)

                self.optimizer.zero_grad()
                loss_td: TensorDictBase = self.loss_tdmodule(sampled_td)
                loss_tensor: Tensor = sum(
                    (loss_td[k] for k in self.loss_keys),
                    torch.tensor(0.0, device=td.device),
                )
                loss_tensor.backward()
                nn.utils.clip_grad_norm_(
                    self.loss_tdmodule.parameters(), max_norm=self.cfg.max_grad_norm
                )
                self.optimizer.step()

                # Accumulate losses
                for k in self.loss_keys:
                    total_loss_dict[k] += loss_td[k].item()

        # Compute average loss
        num_updates = self.cfg.num_epochs * (
            self.batch_size // self.cfg.replay_buffer_batch_size
        )
        process_dict = {k: v / num_updates for k, v in total_loss_dict.items()}

        return process_dict

    @override
    def get_module_device(self) -> torch.device:
        return torch.device(self.module_device)

    @override
    def set_module_device(self, device: torch.device) -> None:
        self.module_device = device

        self.common_tdmodule = self.common_tdmodule.to(self.module_device)
        self.policy_head = self.policy_head.to(self.module_device)
        self.value_head = self.value_head.to(self.module_device)

    @override
    def get_replay_buffer_device(self) -> torch.device:
        return self.replay_buffer_device

    @override
    def set_replay_buffer_device(self, device: torch.device) -> None:
        raise ValueError(
            "set_replay_buffer_device not yet supported for Zannone2019Agent"
        )

    @override
    def get_cheap_info(self) -> dict[str, Any]:
        return {"replay_buffer_count": len(self.replay_buffer)}

    @override
    def get_expensive_info(self) -> dict[str, Any]:
        return {
            # "common_module_norm": module_norm(self.common_module),
            "value_head_norm": module_norm(self.value_head),
            "policy_head_norm": module_norm(self.policy_head),
        }

    # @override
    # def save(self, path: Path) -> None:
    #     path.mkdir(exist_ok=True)
    #
    #     # Store pointnet and encoder as a raw models, weights will be updated either way
    #     torch.save(self.pointnet.to("cpu"), path / "pointnet.pt")
    #     torch.save(self.encoder.to("cpu"), path / "encoder.pt")
    #
    #     # Common module weights
    #     torch.save(
    #         self.common_module.state_dict(),
    #         path / "common_module.pth",
    #     )
    #
    #     # Policy head weights
    #     torch.save(
    #         self.policy_head.state_dict(),
    #         path / "policy_head.pth",
    #     )
    #
    #     # Value head weights
    #     torch.save(
    #         self.value_head.state_dict(),
    #         path / "value_head.pth",
    #     )
    #
    #     # Save agent config
    #     OmegaConf.save(OmegaConf.structured(self.cfg), path / "config.yaml")
    #
    #     # Save the misc args that were passed to the constructor
    #     OmegaConf.save(
    #         OmegaConf.create(
    #             {
    #                 "latent_size": self.latent_size,
    #                 "action_mask_key": self.action_mask_key,
    #                 "batch_size": self.batch_size,
    #             }
    #         ),
    #         path / "args.yaml",
    #     )
    #     torch.save(self.action_spec, path / "action_spec.pt")

    # @override
    # @classmethod
    # def load(
    #     cls: type[Self],
    #     path: Path,
    #     module_device: torch.device,
    #     replay_buffer_device: torch.device,
    # ) -> Self:
    #     # Load agent config
    #     cfg_dict = OmegaConf.merge(
    #         OmegaConf.structured(Zannone2019AgentConfig),
    #         OmegaConf.load(path / "config.yaml"),
    #     )
    #     cfg = cast(Zannone2019AgentConfig, OmegaConf.to_object(cfg_dict))
    #
    #     # Load pointnet and encoder
    #     pointnet: PointNet = torch.load(path / "pointnet.pt")
    #     encoder: nn.Module = torch.load(path / "encoder.pt")
    #
    #     # Load args that were originally passed to the constructor
    #     args = OmegaConf.load(path / "args.yaml")
    #     action_spec = torch.load(path / "action_spec.pt")
    #
    #     # Construct instance of agent
    #     agent = cls(
    #         cfg=cfg,
    #         pointnet=pointnet,
    #         encoder=encoder,
    #         action_spec=action_spec,
    #         latent_size=args.latent_size,
    #         action_mask_key=args.action_mask_key,
    #         batch_size=args.batch_size,
    #         module_device=module_device,
    #         replay_buffer_device=replay_buffer_device,
    #     )
    #
    #     # Load common module weights
    #     agent.common_module.load_state_dict(torch.load(path / "common_module.pth"))
    #     # Load policy head weights
    #     agent.policy_head.load_state_dict(torch.load(path / "policy_head.pth"))
    #     # Load value head weights
    #     agent.value_head.load_state_dict(torch.load(path / "value_head.pth"))
    #
    #     return agent
