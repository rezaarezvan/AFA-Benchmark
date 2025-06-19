from dataclasses import dataclass, field
from typing import Any, final, override
from tensordict.nn import TensorDictModule
from torch import nn
import torch
from torchrl.modules import MLP

from afa_rl.agents import DQNAgent, serializable, unserializable
from afa_rl.shim2018.models import Shim2018Embedder
from afa_rl.utils import get_sequential_module_norm, module_norm
from common.custom_types import FeatureMask, MaskedFeatures


@final
class Shim2018ActionValueNet(nn.Module):
    def __init__(
        self,
        embedder: Shim2018Embedder,
        embedding_size: int,
        action_size: int,
        num_cells: tuple[int, ...],
        dropout: float,
    ):
        super().__init__()
        self.embedder = embedder
        self.embedding_size = embedding_size
        self.action_size = action_size
        self.num_cells = num_cells
        self.dropout = dropout

        self.net = MLP(
            in_features=self.embedding_size,
            out_features=self.action_size,
            num_cells=self.num_cells,
            dropout=self.dropout,
            activation_class=nn.ReLU,
        )

    @override
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


@final
@dataclass(kw_only=True, eq=False, order=False)
class Shim2018Agent(DQNAgent):
    embedder: Shim2018Embedder = unserializable()
    embedding_size: int = serializable(default=int)
    n_features: int = serializable()
    action_value_num_cells: tuple[int, ...] = (128, 128)
    action_value_dropout: float = 0.1
    action_value_net: Shim2018ActionValueNet = field(init=False)

    @override
    def get_action_value_module(self) -> TensorDictModule:
        self.action_value_net = Shim2018ActionValueNet(
            embedder=self.embedder,
            embedding_size=self.embedding_size,
            action_size=self.action_spec.n,
            num_cells=self.action_value_num_cells,
            dropout=self.action_value_dropout,
        )
        action_value_module = TensorDictModule(
            module=self.action_value_net,
            in_keys=["masked_features", "feature_mask", "action_mask"],
            out_keys=["action_value"],
        )
        return action_value_module

    @override
    def get_eval_info(self) -> dict[str, Any]:
        return {
            "value net norm": module_norm(self.action_value_net.net),
            # "embedder norm": module_norm(self.embedder),
        }
