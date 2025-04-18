from tensordict.nn import TensorDictModule
import torch
from tensordict import TensorDict
from torch.distributions import Categorical
from torchrl.data import TensorSpec
from torchrl.envs import ExplorationType, set_exploration_type
from torchrl.modules import MLP, ProbabilisticActor, QValueActor

from afa_rl.agents import Shim2018ValueModule, Zannone2019PolicyModule
from afa_rl.models import PointNet, ReadProcessEncoder, ShimEmbedder, ShimMLPClassifier
from afa_rl.utils import remove_module_prefix
from common.custom_types import AFASelection, FeatureMask, Label, MaskedFeatures


def get_td_from_masked_features(
    masked_features: MaskedFeatures,
    feature_mask: FeatureMask,
) -> TensorDict:
    """
    Creates a TensorDict from the masked features and the feature mask.
    """
    # The action mask is almost the same as the feature mask but with one extra element
    action_mask = torch.ones(
        masked_features.shape[0],
        masked_features.shape[1] + 1,
        dtype=torch.bool,
    )
    action_mask[:, 1:] = feature_mask

    td = TensorDict(
        {
            "action_mask": action_mask,
            "feature_values": masked_features,
            "feature_mask": feature_mask,
        },
        batch_size=masked_features.shape[0],
    )

    return td


class Shim2018AFAMethod:
    """
    Implements the AFAMethod protocol for the Shim2018 agent.
    """

    def __init__(
        self,
        value_network: QValueActor,
        embedder: ShimEmbedder,
        classifier: ShimMLPClassifier,
        action_spec: TensorSpec,
    ):
        self.device = value_network.device
        self.value_network = value_network
        self.embedder = embedder
        self.classifier = classifier
        self.action_spec = action_spec

    def select(
        self, masked_features: MaskedFeatures, feature_mask: FeatureMask
    ) -> AFASelection:
        masked_features = masked_features.to(self.device)
        feature_mask = feature_mask.to(self.device)

        # The agent expects a tensordict with an "embedding" and "action mask" key

        # The embedding is produced by the embedder
        with torch.no_grad():
            embedding = self.embedder(masked_features, feature_mask)

        td = get_td_from_masked_features(masked_features, feature_mask)
        td["embedding"] = embedding

        # Apply the agent's policy to the tensordict
        with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
            td = self.value_network(td)

        # Get the action from the tensordict
        afa_selection = td["action"].unsqueeze(-1)

        return afa_selection

    def predict(
        self, masked_features: MaskedFeatures, feature_mask: FeatureMask
    ) -> Label:
        with torch.no_grad():
            embedding = self.embedder(masked_features, feature_mask)

        logits = self.classifier(embedding)
        probs: Label = logits.softmax(dim=-1)
        return probs

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
                "classifier_config": {
                    "input_size": self.classifier.input_size,
                    "num_classes": self.classifier.num_classes,
                    "num_cells": self.classifier.num_cells,
                },
                "classifier_state_dict": self.classifier.state_dict(),
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
        value_module.eval()

        value_network = QValueActor(
            module=value_module,
            in_keys=["embedding", "action_mask"],
            spec=data["action_spec"],
        )

        encoder = ReadProcessEncoder(**data["encoder_config"])
        encoder.load_state_dict(data["encoder_state_dict"])
        encoder.to(data["device"])
        encoder.eval()

        embedder = ShimEmbedder(encoder)

        classifier = ShimMLPClassifier(**data["classifier_config"])
        classifier.load_state_dict(data["classifier_state_dict"])
        classifier.to(data["device"])
        classifier.eval()

        return Shim2018AFAMethod(
            value_network, embedder, classifier, data["action_spec"]
        )


class Zannone2019AFAMethod:
    """
    Implements the AFAMethod protocol for the Zannone2019 agent.
    """

    def __init__(
        self,
        actor_network: ProbabilisticActor,
    ):
        self.actor_network = actor_network
        self.device = actor_network.device

    def __call__(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
        image_shape: tuple[int, int] | None = None,
    ) -> AFASelection:
        masked_features = masked_features.to(self.device)
        feature_mask = feature_mask.to(self.device)

        td = get_td_from_masked_features(masked_features, feature_mask)

        # Apply the agent's policy to the tensordict
        with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
            td = self.actor_network(td)

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
    def load(path: str) -> "Zannone2019AFAMethod":
        data = torch.load(path, weights_only=False)

        identity_network = MLP(**data["identity_network_config"])
        feature_map_encoder = MLP(**data["feature_map_encoder_config"])

        pointnet = PointNet(
            naive_identity_fn=data["pointnet_config"]["naive_identity_fn"],
            identity_network=identity_network,
            feature_map_encoder=feature_map_encoder,
            pointnet_type=pointnet_type,
        )
        encoder = MLP(
            **data["encoder_config"],
        )

        policy_module = Zannone2019PolicyModule(
            pointnet=pointnet, encoder=encoder, **data["policy_module_config"]
        )

        actor_network = ProbabilisticActor(
            module=TensorDictModule(
                module=policy_module,
                in_keys=["masked_features", "feature_mask", "action_mask"],
                out_keys=["logits"],
            ),
            spec=data["action_spec"],
            in_keys=["logits"],
            distribution_class=Categorical,
            return_log_prob=True,
        )

        return Zannone2019AFAMethod(actor_network)


class RandomDummyAFAMethod:
    """
    A dummy AFAMethod for testing purposes. Chooses a random feature to observe from the masked features.
    """

    def __init__(self, n_classes):
        self.n_classes = n_classes

    def select(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
    ) -> AFASelection:
        """
        Chooses to observe a random feature from the masked features (or stop collecting features).
        """
        # Sample from unobserved features uniformly
        probs = (~feature_mask).float()

        # Avoid division by zero
        row_sums = probs.sum(dim=1, keepdim=True)
        probs = torch.where(
            row_sums > 0, probs / row_sums, probs
        )  # normalize or leave zeros

        # Sample one index per row
        sampled = torch.multinomial(probs, num_samples=1)
        selection = sampled.squeeze(1)  # (B, 1) → (B,)

        # Add 1 to the index to make it 1-based
        selection = selection + 1

        # With 1/n_features probability, select 0 (stop collecting features) for each sample
        stop_collecting_mask = torch.rand(
            selection.shape[0], device=masked_features.device
        ) < (1 / masked_features.shape[1])

        selection[stop_collecting_mask] = 0

        return selection

    def predict(
        self, masked_features: MaskedFeatures, feature_mask: FeatureMask
    ) -> Label:
        """
        Returns a random prediction from the classes.
        """
        # Pick a random class from the classes
        prediction = torch.randint(
            0,
            self.n_classes,
            (masked_features.shape[0],),
            device=masked_features.device,
        )
        # One-hot encode the prediction
        prediction = torch.nn.functional.one_hot(
            prediction, num_classes=self.n_classes
        ).float()

        return prediction

    def save(self, path: str) -> None:
        """
        Saves the method to a file.
        """
        torch.save(
            {
                "n_classes": self.n_classes,
            },
            path,
        )

    @staticmethod
    def load(path: str) -> "RandomDummyAFAMethod":
        """
        Loads the method from a file.
        """
        data = torch.load(path)
        return RandomDummyAFAMethod(data["n_classes"])


class SequentialDummyAFAMethod:
    """
    A dummy AFAMethod for testing purposes. Always chooses the next feature to observe in order.
    """

    def __init__(self, n_classes):
        self.n_classes = n_classes

    def select(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
    ) -> AFASelection:
        # Choose the next unobserved feature
        unobserved_features = (~feature_mask).nonzero(as_tuple=True)[1]
        if unobserved_features.numel() == 0:
            return torch.tensor(0, device=masked_features.device)
        next_feature = unobserved_features[0] + 1
        # With 1/n_features probability, select 0 (stop collecting features) for each sample
        stop_collecting_mask = torch.rand(
            masked_features.shape[0], device=masked_features.device
        ) < (1 / masked_features.shape[1])
        selection = torch.where(
            stop_collecting_mask,
            torch.tensor(0, device=masked_features.device),
            next_feature,
        )
        return selection

    def predict(
        self, masked_features: MaskedFeatures, feature_mask: FeatureMask
    ) -> Label:
        """
        Returns a random prediction from the classes.
        """
        # Pick a random class from the classes
        prediction = torch.randint(
            0,
            self.n_classes,
            (masked_features.shape[0],),
            device=masked_features.device,
        )
        # One-hot encode the prediction
        prediction = torch.nn.functional.one_hot(
            prediction, num_classes=self.n_classes
        ).float()

        return prediction

    def save(self, path: str) -> None:
        """
        Saves the method to a file.
        """
        torch.save(
            {
                "n_classes": self.n_classes,
            },
            path,
        )

    @staticmethod
    def load(path: str) -> "SequentialDummyAFAMethod":
        """
        Loads the method from a file.
        """
        data = torch.load(path)
        return SequentialDummyAFAMethod(data["n_classes"])
