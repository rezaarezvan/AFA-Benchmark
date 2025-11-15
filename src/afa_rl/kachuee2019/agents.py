from typing import Any, final, override

import torch
import torch.nn.functional as F
from tensordict import TensorDictBase
from tensordict.nn import (
    TensorDictModule,
    TensorDictModuleBase,
    TensorDictSequential,
)
from torch import Tensor, nn, optim
from torchrl.data import (
    LazyTensorStorage,
    SamplerWithoutReplacement,
    TensorDictReplayBuffer,
    TensorSpec,
)
from torchrl.modules import EGreedyModule, QValueModule
from torchrl.objectives import DQNLoss, SoftUpdate, ValueEstimators

from afa_rl.agents import Agent
from afa_rl.kachuee2019.models import Kachuee2019PQModule
from afa_rl.utils import module_norm
from common.config_classes import (
    Kachuee2019AgentConfig,
)
from common.custom_types import MaskedFeatures


@final
class Kachuee2019ActionValueModule(nn.Module):
    def __init__(self, pq_module: Kachuee2019PQModule):
        super().__init__()

        self.pq_module = pq_module

    @override
    def forward(self, masked_features: MaskedFeatures, action_mask: Tensor):
        # pq_module.forward ensures that gradients are not backpropagated to the P network
        _class_logits, qvalues = self.pq_module.forward(masked_features)
        # By setting the Q-values of invalid actions to -inf, we prevent them from being selected greedily.
        qvalues[~action_mask] = float("-inf")
        return qvalues


@final
class Kachuee2019Agent(Agent):
    def __init__(
        self,
        action_spec: TensorSpec,
        action_mask_key: str,
        module_device: torch.device,  # device to place nn.Modules on
        replay_buffer_device: torch.device,  # device to place replay buffer on
        pq_module: Kachuee2019PQModule,
        class_weights: Tensor,
        cfg: Kachuee2019AgentConfig,
    ):
        self.action_spec = action_spec
        self.action_mask_key = action_mask_key
        # self.batch_size = batch_size
        self.module_device = module_device
        self.replay_buffer_device = replay_buffer_device
        self.pq_module = pq_module.to(module_device)
        self.class_weights = class_weights.to(module_device)
        self.cfg = cfg

        self.action_value_module = Kachuee2019ActionValueModule(
            pq_module=self.pq_module,
        ).to(self.module_device)

        self.action_value_tdmodule = TensorDictModule(
            module=self.action_value_module,
            in_keys=["masked_features", "action_mask"],
            out_keys=["action_value"],
        )

        self.greedy_tdmodule = QValueModule(
            spec=self.action_spec,
            action_mask_key=self.action_mask_key,
            action_value_key="action_value",
            out_keys=["action", "action_value", "chosen_action_value"],
        ).to(self.module_device)

        self.greedy_policy_tdmodule = TensorDictSequential(
            [self.action_value_tdmodule, self.greedy_tdmodule]
        )

        self.egreedy_tdmodule = EGreedyModule(
            spec=self.action_spec,
            action_key="action",
            action_mask_key=self.action_mask_key,
            annealing_num_steps=self.cfg.eps_annealing_num_batches,
            eps_init=self.cfg.eps_init,
            eps_end=self.cfg.eps_end,
        ).to(self.module_device)

        self.egreedy_policy_tdmodule = TensorDictSequential(
            [self.greedy_policy_tdmodule, self.egreedy_tdmodule]
        )

        self.loss_tdmodule = DQNLoss(
            value_network=self.greedy_policy_tdmodule,
            loss_function=self.cfg.loss_function,
            delay_value=self.cfg.delay_value,
            double_dqn=self.cfg.double_dqn,
            action_space=self.action_spec,
        ).to(self.module_device)

        self.loss_tdmodule.make_value_estimator(
            ValueEstimators.TD0, gamma=self.cfg.gamma
        )

        if self.cfg.delay_value:
            self.target_net_updater = SoftUpdate(
                self.loss_tdmodule, eps=1 - self.cfg.update_tau
            )
        else:
            self.target_net_updater = None

        self.action_value_optimizer = optim.Adam(
            self.loss_tdmodule.parameters(), lr=self.cfg.action_value_lr
        )

        self.classification_optimizer = optim.Adam(
            self.pq_module.parameters(), lr=self.cfg.classification_lr
        )

        self.replay_buffer = TensorDictReplayBuffer(
            storage=LazyTensorStorage(
                max_size=self.cfg.replay_buffer_size,
                device=self.replay_buffer_device,
            ),
            sampler=SamplerWithoutReplacement(),
            batch_size=self.cfg.replay_buffer_batch_size,
        )

    @override
    def get_exploitative_policy(self) -> TensorDictModuleBase:
        return self.greedy_policy_tdmodule

    @override
    def get_exploratory_policy(self) -> TensorDictModuleBase:
        return self.egreedy_policy_tdmodule

    @override
    def get_policy(self) -> TensorDictModuleBase:
        return self.get_exploratory_policy()

    @override
    def get_cheap_info(self) -> dict[str, Any]:
        return {
            "eps": self.egreedy_tdmodule.eps.item(),  # pyright: ignore
            "replay_buffer_count": len(self.replay_buffer),
        }

    @override
    def get_expensive_info(self) -> dict[str, Any]:
        return {
            "p_net_norm": module_norm(self.pq_module.layers_p),
            "q_net_norm": module_norm(self.pq_module.layers_q),
        }

    @override
    def process_batch(self, td: TensorDictBase) -> dict[str, Any]:
        # assert td.batch_size == torch.Size((self.batch_size,)), "Batch size mismatch"

        self.replay_buffer.extend(td)

        # Initialize total loss dictionary
        total_loss_dict = {"qvalue_loss": 0.0, "classification_loss": 0.0}
        td_errors = []

        for _ in range(self.cfg.num_optim):
            sampled_td = self.replay_buffer.sample()
            if self.replay_buffer_device != self.module_device:
                sampled_td = sampled_td.to(self.module_device)

            # Train Q network
            loss_td: TensorDictBase = self.loss_tdmodule(sampled_td)
            loss_tensor: Tensor = loss_td["loss"]
            self.action_value_optimizer.zero_grad()
            loss_tensor.backward()
            nn.utils.clip_grad_norm_(
                self.loss_tdmodule.parameters(),
                max_norm=self.cfg.max_action_value_grad_norm,
            )
            self.action_value_optimizer.step()
            # Update target network
            if self.target_net_updater is not None:
                self.target_net_updater.step()

            # Train classifier
            class_logits_next, _qvalues_next = self.pq_module(
                sampled_td["next", "masked_features"]
            )
            class_loss_next = F.cross_entropy(
                class_logits_next,
                sampled_td["next", "label"],
                weight=self.class_weights,
            ).mean()
            self.classification_optimizer.zero_grad()
            class_loss_next.backward()
            nn.utils.clip_grad_norm_(
                self.pq_module.parameters(),
                max_norm=self.cfg.max_classification_grad_norm,
            )
            self.classification_optimizer.step()

            td_errors.append(sampled_td["td_error"])

            # Accumulate losses
            total_loss_dict["qvalue_loss"] += loss_td["loss"].item()
            total_loss_dict["classification_loss"] += class_loss_next.item()

        # Anneal epsilon for epsilon greedy exploration
        self.egreedy_tdmodule.step()

        # Compute average loss
        process_dict = {
            k: v / self.cfg.num_optim for k, v in total_loss_dict.items()
        }
        process_dict["td_error"] = torch.mean(torch.stack(td_errors)).item()

        return process_dict

    @override
    def get_module_device(self) -> torch.device:
        return self.module_device

    @override
    def set_module_device(self, device: torch.device) -> None:
        self.module_device = device

        # Send modules to device
        self.action_value_module = self.action_value_module.to(
            self.module_device
        )
        self.greedy_tdmodule = self.greedy_tdmodule.to(self.module_device)
        self.egreedy_tdmodule = self.egreedy_tdmodule.to(self.module_device)

    @override
    def get_replay_buffer_device(self) -> torch.device:
        return self.replay_buffer_device

    @override
    def set_replay_buffer_device(self, device: torch.device) -> None:
        raise ValueError(
            "set_replay_buffer_device not yet supported for Shim2018Agent"
        )

    # TODO: implement properly if needed
    # @override
    # def save(self, path: Path) -> None:
    #     path.mkdir(exist_ok=True)
    #
    #     # Store PQ module as a raw model
    #     torch.save(self.pq_module.to("cpu"), path / "pq_module.pt")
    #
    #     # Save agent config
    #     OmegaConf.save(OmegaConf.structured(self.cfg), path / "config.yaml")
    #
    #     # Save the misc args that were passed to the constructor
    #     OmegaConf.save(
    #         OmegaConf.create(
    #             {
    #                 "action_mask_key": self.action_mask_key,
    #                 # "batch_size": self.batch_size,
    #             }
    #         ),
    #         path / "args.yaml",
    #     )
    #     torch.save(self.action_spec, path / "action_spec.pt")

    # TODO: implement properly if needed
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
    #         OmegaConf.structured(Kachuee2019AgentConfig),
    #         OmegaConf.load(path / "config.yaml"),
    #     )
    #     cfg = cast(Kachuee2019AgentConfig, OmegaConf.to_object(cfg_dict))
    #
    #     # Load PQ module
    #     pq_module: Kachuee2019PQModule = torch.load(path / "pq_module.pt")
    #
    #     # Load args that were originally passed to the constructor
    #     args = OmegaConf.load(path / "args.yaml")
    #     action_spec = torch.load(path / "action_spec.pt")
    #
    #     # Construct instance of agent
    #     agent = cls(
    #         action_spec=action_spec,
    #         action_mask_key=args.action_mask_key,
    #         # batch_size=args.batch_size,
    #         module_device=module_device,
    #         replay_buffer_device=replay_buffer_device,
    #         pq_module=pq_module,
    #         cfg=cfg,
    #     )
    #
    #     return agent
