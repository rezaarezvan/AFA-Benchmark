from typing import Callable

import torch
from jaxtyping import Float
from tensordict import TensorDict, TensorDictBase
from torch import Tensor
from torchrl.data import Binary, Categorical, Composite, Unbounded
from torchrl.envs import EnvBase

from afa_rl.custom_types import (
    AFADatasetFn,
    Classifier,
    Embedder,
    Embedding,
    Logits,
)
from common.custom_types import Features, FeatureMask, Label, MaskedFeatures


class AFAMDP(EnvBase):
    batch_locked = False

    def __init__(
        self,
        dataset_fn: AFADatasetFn,  # a function that returns data in batches when called
        embedder: Embedder,  # gives an embedding from features and feature indices
        classifier: Classifier,  # takes an encoding and returns a prediction
        loss_fn: Callable[[Logits, Label], Float[Tensor, "*batch"]],
        acquisition_costs: Float[Tensor, "feature_size"],  # positive values
        invalid_action_cost: float,  # how much to penalize invalid actions
        device: torch.device,
        batch_size: torch.Size,
    ):
        # Do not allow empty batch sizes
        assert batch_size != torch.Size(()), "Batch size must be non-empty"
        super().__init__(device=device, batch_size=batch_size)

        self.dataset_fn = dataset_fn
        self.embedder = embedder
        self.classifier = classifier
        self.loss_fn = loss_fn
        self.acquisition_costs = acquisition_costs
        self.invalid_action_cost = invalid_action_cost

        # Same feature size and label size
        dummy_features, dummy_label = dataset_fn(torch.Size((1,)), move_on=False)
        self.feature_size: int = dummy_features.shape[1]
        self.n_classes: int = dummy_label.shape[1]

        # Calculate the encoding size using dummy sample
        with torch.no_grad():
            dummy_masked_features: Features = torch.zeros(
                torch.Size((1, self.feature_size)),
                dtype=torch.float32,
                device=device,
            )
            dummy_feature_mask: FeatureMask = torch.zeros(
                torch.Size((1, self.feature_size)), dtype=torch.bool, device=device
            )
            dummy_embedding = self.embedder(dummy_masked_features, dummy_feature_mask)
        self.embedding_size: int = dummy_embedding.shape[1]

        self._make_spec()

    def _make_spec(self):
        self.observation_spec = Composite(
            feature_mask=Binary(
                shape=self.batch_size + torch.Size((self.feature_size,)),
                dtype=torch.bool,
            ),
            action_mask=Binary(
                shape=self.batch_size + torch.Size((self.feature_size + 1,)),
                dtype=torch.bool,
            ),
            masked_features=Unbounded(
                shape=self.batch_size + torch.Size((self.feature_size,)),
                dtype=torch.float32,
            ),
            embedding=Unbounded(
                shape=self.batch_size + torch.Size((self.embedding_size,)),
                dtype=torch.float32,
            ),
            model_reward=Unbounded(
                shape=self.batch_size + torch.Size((1,)),
                dtype=torch.float32,
            ),
            fa_reward=Unbounded(
                shape=self.batch_size + torch.Size((1,)),
                dtype=torch.float32,
            ),
            invalid_action_reward=Unbounded(
                shape=self.batch_size + torch.Size((1,)),
                dtype=torch.float32,
            ),
            # hidden from the agent
            features=Unbounded(
                shape=self.batch_size + torch.Size((self.feature_size,)),
                dtype=torch.float32,
            ),
            label=Unbounded(
                shape=self.batch_size + torch.Size((self.n_classes,)),
                dtype=torch.int64,
            ),
            # predicted_label is -1 until agent chooses to end episode
            predicted_class=Unbounded(
                shape=self.batch_size + torch.Size((1,)),
                dtype=torch.int64,
            ),
            batch_size=self.batch_size,
        )
        # action = 0 means stop, action = i means choose feature i
        self.action_spec = Categorical(
            n=self.feature_size + 1,
            shape=self.batch_size + torch.Size(()),
            dtype=torch.int64,
        )
        self.reward_spec = Unbounded(
            shape=self.batch_size + torch.Size((1,)), dtype=torch.float32
        )

    def _reset(self, tensordict: TensorDictBase, **_):
        if tensordict is None:
            tensordict = TensorDict({}, batch_size=self.batch_size, device=self.device)

        # Get a sample from the dataset
        features, label = self.dataset_fn(tensordict.batch_size)
        features: Features = features.to(tensordict.device)
        label: Label = label.to(tensordict.device)
        # The indices of chosen features so far, start with no features
        feature_mask: FeatureMask = torch.zeros_like(
            features, dtype=torch.bool, device=tensordict.device
        )
        # The values of chosen features so far, start with no features
        masked_features: MaskedFeatures = torch.zeros_like(
            features, dtype=torch.float32, device=tensordict.device
        )
        embedding = self.embedder(masked_features, feature_mask)
        model_reward = torch.zeros(
            tensordict.batch_size + (1,), dtype=torch.float32, device=tensordict.device
        )
        fa_reward = torch.zeros(
            tensordict.batch_size + (1,), dtype=torch.float32, device=tensordict.device
        )
        invalid_action_reward = torch.zeros(
            tensordict.batch_size + (1,), dtype=torch.float32, device=tensordict.device
        )

        td = TensorDict(
            {
                "action_mask": torch.ones(
                    tensordict.batch_size + (self.feature_size + 1,),
                    dtype=torch.bool,
                    device=tensordict.device,
                ),
                "feature_mask": feature_mask,
                "masked_features": masked_features,
                "embedding": embedding,
                "model_reward": model_reward,
                "fa_reward": fa_reward,
                "invalid_action_reward": invalid_action_reward,
                "features": features,
                "label": label,
                "predicted_class": -1
                * torch.ones(
                    tensordict.batch_size + (1,),
                    dtype=torch.int64,
                    device=tensordict.device,
                ),
            },
            batch_size=tensordict.batch_size,
            device=tensordict.device,
        )
        return td

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        new_feature_mask: FeatureMask = tensordict["feature_mask"].clone()
        new_masked_features: MaskedFeatures = tensordict["masked_features"].clone()
        new_embedding: Embedding = tensordict["embedding"].clone()
        new_predicted_class = tensordict["predicted_class"].clone()
        new_action_mask = tensordict["action_mask"].clone()

        # Process stopping case. We don't have to compute new features and embeddings since the features don't change
        is_stop = tensordict["action"] == 0
        model_output = self.classifier(tensordict["embedding"][is_stop])
        loss = self.loss_fn(model_output, tensordict["label"][is_stop])
        model_reward = torch.zeros(
            tensordict.batch_size + (1,), dtype=torch.float32, device=tensordict.device
        )
        model_reward[is_stop] = -loss.unsqueeze(-1)
        new_predicted_class[is_stop] = model_output.argmax(dim=-1, keepdim=True)

        # Process feature acquisition case, compute new features and embeddings
        is_fa = tensordict["action"] != 0
        new_feature_indices = tensordict["action"][is_fa] - 1
        new_feature_mask[is_fa, new_feature_indices] = True
        new_masked_features[is_fa, new_feature_indices] = tensordict["features"][
            is_fa, new_feature_indices
        ].clone()
        new_embedding[is_fa] = self.embedder(
            new_masked_features[is_fa], new_feature_mask[is_fa]
        )
        fa_reward = torch.zeros(
            tensordict.batch_size + (1,), dtype=torch.float32, device=tensordict.device
        )
        fa_reward[is_fa] = -self.acquisition_costs[new_feature_indices].unsqueeze(-1)
        # Update action_mask
        new_action_mask[is_fa, new_feature_indices + 1] = False

        # Penalize invalid actions by looking at action and action_mask
        invalid_action_reward = torch.zeros(
            tensordict.batch_size + (1,), dtype=torch.float32, device=tensordict.device
        )
        is_invalid = ~tensordict["action_mask"][
            torch.arange(len(tensordict)), tensordict["action"]
        ]
        invalid_action_reward[is_invalid] = -self.invalid_action_cost

        reward = model_reward + fa_reward + invalid_action_reward

        # Compute done mask
        done = is_stop.unsqueeze(-1)

        return TensorDict(
            {
                "action_mask": new_action_mask,
                "feature_mask": new_feature_mask,
                "masked_features": new_masked_features,
                "embedding": new_embedding,
                "fa_reward": fa_reward,
                "model_reward": model_reward,
                "invalid_action_reward": invalid_action_reward,
                # features and label are not cloned since they stay the same
                "features": tensordict["features"],
                "label": tensordict["label"],
                "predicted_class": new_predicted_class,
                "done": done,
                "reward": reward,
            },
            batch_size=tensordict.batch_size,
        )

    def _set_seed(self, seed: int | None):
        rng = torch.manual_seed(seed)  # type: ignore
        self.rng = rng
