from typing import Callable

import torch
from jaxtyping import Float
from tensordict import TensorDict, TensorDictBase
from torch import Tensor
from torchrl.data import Binary, Categorical, Composite, Unbounded
from torchrl.envs import EnvBase

from afa_rl.custom_types import (
    DatasetFn,
    Embedder,
    Embedding,
    Feature,
    FeatureMask,
    TaskLabel,
    TaskModel,
    TaskModelOutput,
)


class AFAMDP(EnvBase):
    def __init__(
        self,
        dataset_fn: DatasetFn,  # a function that returns data in batches when called
        embedder: Embedder,  # gives an embedding from features and feature indices
        task_model: TaskModel,  # takes an encoding and returns a prediction
        loss_fn: Callable[[TaskModelOutput, TaskLabel], Float[Tensor, "*batch"]],
        acquisition_costs: Float[Tensor, "feature_size"],
        device: torch.device,
        batch_size: torch.Size,
    ):
        super().__init__(device=device, batch_size=batch_size)

        self.dataset_fn = dataset_fn
        self.embedder = embedder
        self.task_model = task_model
        self.loss_fn = loss_fn
        self.acquisition_costs = acquisition_costs

        # Same feature size and label size
        dummy_sample = dataset_fn(torch.Size((1,)), move_on=False)
        self.feature_size: int = dummy_sample.feature.shape[1]
        self.label_size: int = dummy_sample.label.shape[1]
        # Labels can in general have different dtypes
        self.label_dtype = dummy_sample.label.dtype

        # Calculate the encoding size using dummy sample
        with torch.no_grad():
            dummy_feature_values: Feature = torch.zeros(
                torch.Size((1, self.feature_size)),
                dtype=torch.float32,
                device=device,
            )
            dummy_feature_mask: FeatureMask = torch.zeros(
                torch.Size((1, self.feature_size)), dtype=torch.bool, device=device
            )
            dummy_embedding = self.embedder(dummy_feature_values, dummy_feature_mask)
        self.embedding_size: int = dummy_embedding.shape[1]

        self._make_spec()

    def _make_spec(self):
        self.observation_spec = Composite(
            feature_mask=Binary(
                shape=self.batch_size + (self.feature_size,),
                dtype=torch.bool,
            ),
            # Which actions the agent is allowed to take
            action_mask=Binary(
                shape=self.batch_size + (self.feature_size + 1,),
                dtype=torch.bool,
            ),
            feature_values=Unbounded(
                shape=self.batch_size + (self.feature_size,),
                dtype=torch.float32,
            ),
            embedding=Unbounded(
                shape=self.batch_size + (self.embedding_size,),
                dtype=torch.float32,
            ),
            model_reward=Unbounded(
                shape=self.batch_size + (1,),
                dtype=torch.float32,
            ),
            fa_reward=Unbounded(
                shape=self.batch_size + (1,),
                dtype=torch.float32,
            ),
            # hidden from the agent
            all_features=Unbounded(
                shape=self.batch_size + (self.feature_size,),
                dtype=torch.float32,
            ),
            label=Unbounded(
                shape=self.batch_size + (self.label_size,), dtype=self.label_dtype
            ),
            batch_size=self.batch_size,
        )
        # action = 0 means stop, action = i means choose feature i
        self.action_spec = Categorical(
            n=self.feature_size + 1,
            shape=self.batch_size,
            dtype=torch.int64,
        )
        self.reward_spec = Unbounded(
            shape=self.batch_size + torch.Size((1,)), dtype=torch.float32
        )

    def _reset(self, tensordict: TensorDictBase, **_):
        if tensordict is not None:
            reset_batch_size = tensordict.shape

        else:
            reset_batch_size = self.batch_size
        # Get a sample from the dataset
        sample = self.dataset_fn(reset_batch_size)
        all_features: Feature = sample.feature.to(self.device)
        label: TaskLabel = sample.label.to(self.device)
        # the indices of chosen features so far, start with no features
        feature_mask: FeatureMask = torch.zeros_like(
            all_features, dtype=torch.bool, device=self.device
        )
        # the values of chosen features so far, start with no features
        feature_values: Feature = torch.zeros_like(
            all_features, dtype=torch.float32, device=self.device
        )
        embedding = self.embedder(feature_values, feature_mask)
        model_reward = torch.zeros(
            reset_batch_size + (1,), dtype=torch.float32, device=self.device
        )
        fa_reward = torch.zeros(
            reset_batch_size + (1,), dtype=torch.float32, device=self.device
        )
        # Agent is allowed to choose any feature at the beginning
        action_mask = torch.ones(
            reset_batch_size + (self.feature_size + 1,),
            dtype=torch.bool,
            device=self.device,
        )

        if tensordict is not None:
            tensordict["feature_mask"] = feature_mask
            tensordict["feature_values"] = feature_values
            tensordict["embedding"] = embedding
            tensordict["model_reward"] = model_reward
            tensordict["fa_reward"] = fa_reward
            tensordict["all_features"] = all_features
            tensordict["label"] = label
            tensordict["action_mask"] = action_mask
            return tensordict
        else:
            return TensorDict(
                {
                    "feature_mask": feature_mask,
                    "feature_values": feature_values,
                    "embedding": embedding,
                    "model_reward": model_reward,
                    "fa_reward": fa_reward,
                    "all_features": all_features,
                    "label": label,
                    "action_mask": action_mask,
                },
                batch_size=self.batch_size,
            )

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        # Clone feature data and embedding data. This will change for samples where AFA is done
        feature_mask: Feature = tensordict["feature_mask"].clone()
        feature_values: FeatureMask = tensordict["feature_values"].clone()
        embedding: Embedding = tensordict["embedding"].clone()
        action_mask = tensordict["action_mask"].clone()

        # Process stopping case. We don't have to compute new features and embeddings since the features don't change
        is_stop = tensordict["action"] == 0
        # TODO: for debugging
        if is_stop.any():
            pass
        model_output = self.task_model(tensordict["embedding"][is_stop])
        loss = self.loss_fn(model_output, tensordict["label"][is_stop])
        model_reward = torch.zeros(
            self.batch_size + (1,), dtype=torch.float32, device=self.device
        )
        model_reward[is_stop] = -loss.unsqueeze(-1)
        action_mask[is_stop, 0] = False

        # Process feature acquisition case, compute new features and embeddings
        is_fa = tensordict["action"] != 0
        new_feature_indices = tensordict["action"][is_fa] - 1
        feature_mask[is_fa, new_feature_indices] = True
        feature_values[is_fa, new_feature_indices] = tensordict["all_features"][
            is_fa, new_feature_indices
        ].clone()
        embedding[is_fa] = self.embedder(feature_values[is_fa], feature_mask[is_fa])
        action_mask[is_fa, new_feature_indices + 1] = False

        # Compute rewards using conditional selection
        fa_reward = torch.zeros(
            self.batch_size + (1,), dtype=torch.float32, device=self.device
        )
        fa_reward[is_fa] = self.acquisition_costs[new_feature_indices].unsqueeze(-1)

        reward = model_reward + fa_reward

        # Compute done mask
        done = is_stop.unsqueeze(-1)

        return TensorDict(
            {
                "feature_mask": feature_mask,
                "feature_values": feature_values,
                "embedding": embedding,
                "fa_reward": fa_reward,
                "model_reward": model_reward,
                # TODO: maybe needs cloning here
                "all_features": tensordict["all_features"],
                "label": tensordict["label"],
                "done": done,
                "reward": reward,
                "action_mask": action_mask,
            },
            batch_size=self.batch_size,
        )

    def _set_seed(self, seed: int | None):
        rng = torch.manual_seed(seed)  # type: ignore
        self.rng = rng
