
from dataclasses import dataclass
from typing import Any
from tensordict.nn import TensorDictModule
from torch import nn
import torch
from torchrl.modules import MLP

from afa_rl.agents import DQNAgent, serializable, unserializable
from afa_rl.shim2018.models import Shim2018Embedder
from afa_rl.utils import get_sequential_module_norm, module_norm
from common.custom_types import FeatureMask, MaskedFeatures

class Shim2018ActionValueNet(nn.Module):
    def __init__(
        self, embedder: Shim2018Embedder, embedding_size, action_size, num_cells
    ):
        super().__init__()
        self.embedder = embedder
        self.embedding_size = embedding_size
        self.action_size = action_size
        self.num_cells = num_cells

        self.net = MLP(
            in_features=embedding_size,
            # in_features=20+embedding_size,
            out_features=action_size,
            num_cells=num_cells,
            activation_class=nn.ReLU,
        )

    def forward(
        self, masked_features: MaskedFeatures, feature_mask: FeatureMask, action_mask
    ):
        # We do not want to update the embedder weights using the Q-values, this is done separately in the training loop
        with torch.no_grad():
            embedding = self.embedder(masked_features, feature_mask)
        qvalues = self.net(embedding)
        # qvalues = self.net(feature_mask.float())
        # qvalues = self.net(torch.cat([embedding, torch.zeros_like(masked_features)], dim=-1))
        # By setting the Q-values of invalid actions to -inf, we prevent them from being selected greedily.
        qvalues[~action_mask] = float("-inf")
        return qvalues

# class DummyShim2018ActionValueNet(nn.Module):
#     def __init__(
#         self, n_features: int, action_size: int, num_cells
#     ):
#         super().__init__()
#         self.n_features = n_features
#         self.action_size = action_size

#         self.net = MLP(
#             in_features=n_features*2,
#             out_features=action_size,
#             num_cells=num_cells,
#             activation_class=nn.ReLU,
#         )
#         # self.net = nn.Sequential(
#         #     nn.Linear(2*n_features, 128),
#         #     nn.ReLU(),
#         #     nn.Linear(128, 128),
#         #     nn.ReLU(),
#         #     nn.Linear(128, action_size),
#         # )

#     def forward(
#         self, masked_features: MaskedFeatures, feature_mask: FeatureMask, action_mask
#     ):
#         qvalues = self.net(torch.cat([masked_features, feature_mask.float()], dim=-1))
#         qvalues[~action_mask] = float("-inf")
#         return qvalues


@dataclass(kw_only=True, eq=False, order=False)
class Shim2018Agent(DQNAgent):
    embedder: Shim2018Embedder = unserializable()
    embedding_size: int = serializable(default=int)
    n_features: int = serializable()

    def get_action_value_module(self) -> TensorDictModule:
        self.action_value_net = Shim2018ActionValueNet(
            embedder=self.embedder,
            embedding_size=self.embedding_size,
            action_size=self.action_spec.n,
            num_cells=[32, 32],
        )
        # self.action_value_net = DummyShim2018ActionValueNet(
        #     n_features=self.n_features,
        #     action_size=self.action_spec.n,
        #     num_cells=[32, 32],
        # )
        action_value_module = TensorDictModule(
            module=self.action_value_net,
            in_keys=["masked_features", "feature_mask", "action_mask"],
            out_keys=["action_value"]
        )
        return action_value_module

    def get_eval_info(self) -> dict[str, Any]:
        return {
            "value net norm": module_norm(self.action_value_net.net),
            # "embedder norm": module_norm(self.embedder),
        }
