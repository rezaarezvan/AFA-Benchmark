from typing import Callable
from jaxtyping import Bool
import torch.nn.functional as F

import torch
from jaxtyping import Float
from tensordict import TensorDict, TensorDictBase
from torch import Tensor
from torch import nn
from torchrl.data import Binary, Categorical, Composite, Unbounded
from torchrl.envs import EnvBase

from afa_rl.custom_types import (
    AFAClassifier,
    AFADatasetFn,
    Classifier,
    Embedder,
    Embedding,
    Logits,
)
from afa_rl.models import PartialVAE, ShimEmbedder, ShimMLPClassifier
from common.custom_types import (
    AFAReward,
    AFARewardFn,
    AFASelection,
    FeatureMask,
    Features,
    Label,
    MaskedFeatures,
)



class AFAEnv(EnvBase):
    """
    A fixed-length MDP for active feature acquisition (AFA).

    It assumes that the agent can choose to acquire features `hard_budget` times.
    """

    batch_locked = False

    def __init__(
        self,
        dataset_fn: AFADatasetFn,  # a function that returns data in batches when called
        reward_fn: AFARewardFn,
        device: torch.device,
        batch_size: torch.Size,
        feature_size: int,
        n_classes: int,
        hard_budget: int # how many features can be acquired before the episode ends
    ):
        # Do not allow empty batch sizes
        assert batch_size != torch.Size(()), "Batch size must be non-empty"
        assert len(batch_size) == 1, "Batch size must be 1D"
        super().__init__(device=device, batch_size=batch_size)

        self.dataset_fn = dataset_fn
        self.reward_fn = reward_fn
        self.feature_size = feature_size
        self.n_classes = n_classes
        self.hard_budget = hard_budget

        self._make_spec()

    def _make_spec(self):
        self.observation_spec = Composite(
            feature_mask=Binary(
                shape=self.batch_size + torch.Size((self.feature_size,)),
                dtype=torch.bool,
            ),
            action_mask=Binary(
                shape=self.batch_size + torch.Size((self.feature_size,)),
                dtype=torch.bool,
            ),
            masked_features=Unbounded(
                shape=self.batch_size + torch.Size((self.feature_size,)),
                dtype=torch.float32,
            ),
            # hidden from the agent
            features=Unbounded(
                shape=self.batch_size + torch.Size((self.feature_size,)),
                dtype=torch.float32,
            ),
            label=Unbounded(
                shape=self.batch_size + torch.Size((self.n_classes,)),
                dtype=torch.float32,
            ),
            batch_size=self.batch_size,
        )
        # One action per feature
        self.action_spec = Categorical(
            n=self.feature_size,
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

        td = TensorDict(
            {
                "action_mask": torch.ones(
                    tensordict.batch_size + (self.feature_size,),
                    dtype=torch.bool,
                    device=tensordict.device,
                ),
                "feature_mask": feature_mask,
                "masked_features": masked_features,
                "features": features,
                "label": label,
            },
            batch_size=tensordict.batch_size,
            device=tensordict.device,
        )
        return td

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        new_feature_mask: FeatureMask = tensordict["feature_mask"].clone()
        new_masked_features: MaskedFeatures = tensordict["masked_features"].clone()
        new_action_mask = tensordict["action_mask"].clone()

        batch_numel = tensordict.batch_size.numel()
        batch_idx = torch.arange(batch_numel, device=tensordict.device)

        # Acquire new features
        new_feature_mask[batch_idx, tensordict["action"]] = True
        new_masked_features[batch_idx, tensordict["action"]] = tensordict["features"][batch_idx, tensordict["action"]].clone()

        # Update action_mask
        new_action_mask[batch_idx, tensordict["action"]] = False

        # Done if we exceed the hard budget
        done = tensordict["feature_mask"].sum(-1) >= self.hard_budget

        # Always calculate a possible reward
        reward = self.reward_fn(
            tensordict["masked_features"],
            tensordict["feature_mask"],
            new_masked_features,
            new_feature_mask,
            tensordict["action"],
            tensordict["features"],
            tensordict["label"],
            done,
        )

        return TensorDict(
            {
                "action_mask": new_action_mask,
                "feature_mask": new_feature_mask,
                "masked_features": new_masked_features,
                "done": done,
                "reward": reward,
                # features and label are not cloned since they stay the same
                "features": tensordict["features"],
                "label": tensordict["label"],
            },
            batch_size=tensordict.batch_size,
        )

    def _set_seed(self, seed: int | None):
        rng = torch.manual_seed(seed)  # type: ignore
        self.rng = rng

def get_common_reward_fn(
    classifier: AFAClassifier,
    loss_fn: Callable[[Logits, Label], AFAReward]
) -> AFARewardFn:
    """
    A standard AFA-RL reward function where the only reward the agent receives is the negative
    classification loss at the end.
    """

    def f(
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
        new_masked_features: MaskedFeatures,
        new_feature_mask: FeatureMask,
        afa_selection: AFASelection,
        features: Features,
        label: Label,
        done: Bool[Tensor, "*batch 1"],
    ) -> AFAReward:
        reward = torch.zeros_like(afa_selection, dtype=torch.float32)

        # If AFA stops, reward is negative loss
        logits = classifier(masked_features[done], feature_mask[done])
        reward[done] = -loss_fn(
            logits,
            label[done],
        )

        return reward

    return f
