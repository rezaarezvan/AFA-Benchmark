import torch
from tensordict import TensorDict
from torchrl.data import TensorSpec
from torchrl.modules import QValueActor

from afa_rl.agents import Shim2018ValueModule
from afa_rl.models import ReadProcessEncoder, ShimEmbedder
from afa_rl.utils import remove_module_prefix
from common.custom_types import AFASelection, FeatureMask, MaskedFeatures


class Shim2018AFAMethod:
    """
    Implements the AFAMethod protocol for the Shim2018 agent.
    """

    def __init__(
        self,
        value_network: QValueActor,
        embedder: ShimEmbedder,
        action_spec: TensorSpec,
    ):
        assert value_network.device == embedder.encoder.device, (
            f"Value module and embedder must be on the same device, instead got {value_network.device} and {embedder.encoder.device} of types {type(value_network.device)} and {type(embedder.encoder.device)}"
        )
        self.device = value_network.device
        self.value_network = value_network
        self.embedder = embedder
        self.action_spec = action_spec

    def __call__(
        self, masked_features: MaskedFeatures, feature_mask: FeatureMask
    ) -> AFASelection:
        masked_features = masked_features.to(self.device)
        feature_mask = feature_mask.to(self.device)

        # The agent expects a tensordict with an "embedding" and "action mask" key

        # The embedding is produced by the embedder
        embedding = self.embedder(masked_features, feature_mask)

        # The action mask is almost the same as the feature mask but with one extra element
        action_mask = torch.ones(
            masked_features.shape[0],
            masked_features.shape[1] + 1,
            dtype=torch.bool,
            device=self.device,
        )
        action_mask[:, 1:] = feature_mask

        td = TensorDict(
            {
                "embedding": embedding,
                "action_mask": action_mask,
                "feature_values": masked_features,
                "feature_mask": feature_mask,
            },
            batch_size=masked_features.shape[0],
            device=self.device,
        )

        # Apply the agent's policy to the tensordict
        td = self.value_network(td)

        # Get the action from the tensordict
        afa_selection = td["action"].unsqueeze(-1)

        return afa_selection

    def save(self, path: str):
        # We have to save two neural networks: the value module and the encoder
        torch.save(
            {
                "action_spec": self.action_spec,
                "value_module_config": {
                    "embedding_size": self.value_network.module[0].embedding_size,
                    "action_size": self.value_network.module[0].action_size,
                    "num_cells": self.value_network.module[0].num_cells,
                },
                "value_module_state_dict": remove_module_prefix(
                    self.value_network.module[0].state_dict()
                ),
                "encoder_config": {
                    "feature_size": self.embedder.encoder.feature_size,
                    "output_size": self.embedder.encoder.output_size,
                    "reading_block_cells": self.embedder.encoder.reading_block_cells,
                    "writing_block_cells": self.embedder.encoder.writing_block_cells,
                    "memory_size": self.embedder.encoder.memory_size,
                    "processing_steps": self.embedder.encoder.processing_steps,
                },
                "encoder_state_dict": self.embedder.encoder.state_dict(),
                "device": str(self.device),
            },
            path,
        )

    @staticmethod
    def load(path: str) -> "Shim2018AFAMethod":
        # We have to load two neural networks: the value module and the encoder
        data = torch.load(path, weights_only=False)

        value_module = Shim2018ValueModule(
            **data["value_module_config"], device=data["device"]
        )
        value_module.load_state_dict(data["value_module_state_dict"])
        value_module.to(data["device"])

        value_network = QValueActor(
            module=value_module,
            in_keys=["embedding", "action_mask"],
            spec=data["action_spec"],
        )

        encoder = ReadProcessEncoder(**data["encoder_config"])
        encoder.load_state_dict(data["encoder_state_dict"])
        encoder.to(data["device"])

        embedder = ShimEmbedder(encoder)

        return Shim2018AFAMethod(value_network, embedder, data["action_spec"])
