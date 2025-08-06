from unittest import TestCase

import torch
from torch import nn

from afa_rl.afa_env import Shim2018Env
from afa_rl.custom_types import (
    AFADatasetFn,
    Classifier,
    Embedder,
    Embedding,
    Logits,
)
from afa_rl.datasets import get_afa_dataset_fn
from afa_rl.utils import floatwrapfn
from common.custom_types import FeatureMask, MaskedFeatures


def get_dummy_data_fn() -> AFADatasetFn:
    """A very simple dataset for debugging
    """
    features = torch.tensor(
        [
            [1, 2, 3, 4, 5],
            [2, 3, 4, 5, 6],
            [3, 4, 5, 6, 7],
            [4, 5, 6, 7, 8],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=torch.int64,
    )

    return get_afa_dataset_fn(features, labels)


class LinearEncoder(Embedder):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.net = nn.Linear(input_size, output_size)

    def forward(
        self, masked_features: MaskedFeatures, feature_mask: FeatureMask
    ) -> Embedding:
        return self.net(torch.cat([masked_features, feature_mask], dim=-1))


class LinearTaskModel(Classifier):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.net = nn.Linear(input_size, output_size)

    def forward(self, embedding: Embedding) -> Logits:
        return self.net(embedding)


class TestAFAMDP(TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.env = Shim2018Env(
            dataset_fn=get_dummy_data_fn(),
            embedder=LinearEncoder(10, 5),
            classifier=LinearTaskModel(5, 4),
            loss_fn=floatwrapfn(nn.CrossEntropyLoss(reduction="none")),
            acquisition_costs=torch.tensor([10, 11, 12, 13, 14], dtype=torch.float32),
            invalid_action_cost=42,
            device=self.device,
            batch_size=torch.Size((2,)),
        )
        # check_env_specs(self.env)
        self.td = self.env.reset()

    def get_state1(self):
        td = self.td.clone()
        td["action"] = torch.tensor([1, 2], dtype=torch.int64, device=self.device)
        td = self.env.step(td)
        return td["next"]

    def get_state2(self):
        td = self.get_state1()
        # pick an invalid action
        td["action"] = torch.tensor([3, 2], dtype=torch.int64, device=self.device)
        td = self.env.step(td)
        return td["next"]

    def get_state3(self):
        td = self.get_state2()
        # pick last feature for first sample, stop for second sample
        td["action"] = torch.tensor([5, 0], dtype=torch.int64, device=self.device)
        td = self.env.step(td)
        return td["next"]

    def get_state3_with_reset(self):
        td = self.get_state2()
        td["action"] = torch.tensor([5, 0], dtype=torch.int64, device=self.device)
        _, td = self.env.step_and_maybe_reset(td)
        return td

    def test_initial_state(self):
        assert self.td.batch_size == torch.Size((2,))
        torch.testing.assert_close(
            self.td["action_mask"],
            torch.tensor(
                [
                    [1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1],
                ],
                dtype=torch.bool,
                device=self.device,
            ),
        )
        torch.testing.assert_close(
            self.td["feature_mask"],
            torch.tensor(
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                ],
                dtype=torch.bool,
                device=self.device,
            ),
        )
        torch.testing.assert_close(
            self.td["masked_features"],
            torch.tensor(
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                ],
                dtype=torch.float32,
                device=self.device,
            ),
        )
        torch.testing.assert_close(
            self.td["fa_reward"],
            torch.tensor(
                [
                    [0],
                    [0],
                ],
                dtype=torch.float32,
                device=self.device,
            ),
        )
        torch.testing.assert_close(
            self.td["model_reward"],
            torch.tensor(
                [
                    [0],
                    [0],
                ],
                dtype=torch.float32,
                device=self.device,
            ),
        )
        torch.testing.assert_close(
            self.td["invalid_action_reward"],
            torch.tensor(
                [
                    [0],
                    [0],
                ],
                dtype=torch.float32,
                device=self.device,
            ),
        )
        torch.testing.assert_close(
            self.td["features"],
            torch.tensor(
                [
                    [1, 2, 3, 4, 5],
                    [2, 3, 4, 5, 6],
                ],
                dtype=torch.float32,
                device=self.device,
            ),
        )
        torch.testing.assert_close(
            self.td["label"],
            torch.tensor(
                [
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                ],
                dtype=torch.int64,
                device=self.device,
            ),
        )
        torch.testing.assert_close(
            self.td["predicted_class"],
            torch.tensor(
                [
                    [-1],
                    [-1],
                ],
                dtype=torch.int64,
                device=self.device,
            ),
        )

    def test_state1(self):
        td = self.get_state1()
        assert td.batch_size == torch.Size((2,))
        torch.testing.assert_close(
            td["action_mask"],
            torch.tensor(
                [
                    [1, 0, 1, 1, 1, 1],
                    [1, 1, 0, 1, 1, 1],
                ],
                dtype=torch.bool,
                device=self.device,
            ),
        )
        torch.testing.assert_close(
            td["feature_mask"],
            torch.tensor(
                [
                    [1, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0],
                ],
                dtype=torch.bool,
                device=self.device,
            ),
        )
        torch.testing.assert_close(
            td["masked_features"],
            torch.tensor(
                [
                    [1, 0, 0, 0, 0],
                    [0, 3, 0, 0, 0],
                ],
                dtype=torch.float32,
                device=self.device,
            ),
        )
        torch.testing.assert_close(
            td["fa_reward"],
            torch.tensor(
                [
                    [-10],
                    [-11],
                ],
                dtype=torch.float32,
                device=self.device,
            ),
        )
        torch.testing.assert_close(
            td["model_reward"],
            torch.tensor(
                [
                    [0],
                    [0],
                ],
                dtype=torch.float32,
                device=self.device,
            ),
        )
        torch.testing.assert_close(
            td["invalid_action_reward"],
            torch.tensor(
                [
                    [0],
                    [0],
                ],
                dtype=torch.float32,
                device=self.device,
            ),
        )
        torch.testing.assert_close(
            td["features"],
            torch.tensor(
                [
                    [1, 2, 3, 4, 5],
                    [2, 3, 4, 5, 6],
                ],
                dtype=torch.float32,
                device=self.device,
            ),
        )
        torch.testing.assert_close(
            td["label"],
            torch.tensor(
                [
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                ],
                dtype=torch.int64,
                device=self.device,
            ),
        )
        torch.testing.assert_close(
            self.td["predicted_class"],
            torch.tensor(
                [
                    [-1],
                    [-1],
                ],
                dtype=torch.int64,
                device=self.device,
            ),
        )
        torch.testing.assert_close(
            td["done"],
            torch.tensor(
                [
                    [0],
                    [0],
                ],
                dtype=torch.bool,
                device=self.device,
            ),
        )

    def test_state2(self):
        td = self.get_state2()
        assert td.batch_size == torch.Size((2,))
        torch.testing.assert_close(
            td["action_mask"],
            torch.tensor(
                [
                    [1, 0, 1, 0, 1, 1],
                    [1, 1, 0, 1, 1, 1],
                ],
                dtype=torch.bool,
                device=self.device,
            ),
        )
        torch.testing.assert_close(
            td["feature_mask"],
            torch.tensor(
                [
                    [1, 0, 1, 0, 0],
                    [0, 1, 0, 0, 0],
                ],
                dtype=torch.bool,
                device=self.device,
            ),
        )
        torch.testing.assert_close(
            td["masked_features"],
            torch.tensor(
                [
                    [1, 0, 3, 0, 0],
                    [0, 3, 0, 0, 0],
                ],
                dtype=torch.float32,
                device=self.device,
            ),
        )
        torch.testing.assert_close(
            td["fa_reward"],
            torch.tensor(
                [
                    [-12],
                    [-11],
                ],
                dtype=torch.float32,
                device=self.device,
            ),
        )
        torch.testing.assert_close(
            td["model_reward"],
            torch.tensor(
                [
                    [0],
                    [0],
                ],
                dtype=torch.float32,
                device=self.device,
            ),
        )
        torch.testing.assert_close(
            td["invalid_action_reward"],
            torch.tensor(
                [
                    [0],
                    [-42],
                ],
                dtype=torch.float32,
                device=self.device,
            ),
        )
        torch.testing.assert_close(
            td["features"],
            torch.tensor(
                [
                    [1, 2, 3, 4, 5],
                    [2, 3, 4, 5, 6],
                ],
                dtype=torch.float32,
                device=self.device,
            ),
        )
        torch.testing.assert_close(
            td["label"],
            torch.tensor(
                [
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                ],
                dtype=torch.int64,
                device=self.device,
            ),
        )
        torch.testing.assert_close(
            self.td["predicted_class"],
            torch.tensor(
                [
                    [-1],
                    [-1],
                ],
                dtype=torch.int64,
                device=self.device,
            ),
        )
        torch.testing.assert_close(
            td["done"],
            torch.tensor(
                [
                    [0],
                    [0],
                ],
                dtype=torch.bool,
                device=self.device,
            ),
        )

    def test_state3(self):
        td = self.get_state3()
        assert td.batch_size == torch.Size((2,))
        torch.testing.assert_close(
            td["action_mask"],
            torch.tensor(
                [
                    [1, 0, 1, 0, 1, 0],
                    [1, 1, 0, 1, 1, 1],
                ],
                dtype=torch.bool,
                device=self.device,
            ),
        )
        torch.testing.assert_close(
            td["feature_mask"],
            torch.tensor(
                [
                    [1, 0, 1, 0, 1],
                    [0, 1, 0, 0, 0],
                ],
                dtype=torch.bool,
                device=self.device,
            ),
        )
        torch.testing.assert_close(
            td["masked_features"],
            torch.tensor(
                [
                    [1, 0, 3, 0, 5],
                    [0, 3, 0, 0, 0],
                ],
                dtype=torch.float32,
                device=self.device,
            ),
        )
        torch.testing.assert_close(
            td["fa_reward"],
            torch.tensor(
                [
                    [-14],
                    [0],
                ],
                dtype=torch.float32,
                device=self.device,
            ),
        )
        # only check that the first sample does not have a model reward
        torch.testing.assert_close(
            td["model_reward"][0],
            torch.tensor(
                [0],
                dtype=torch.float32,
                device=self.device,
            ),
        )
        # the second sample should have a non-zero model reward
        assert torch.any(td["model_reward"][1] != 0), "Expected a non-zero model reward"
        torch.testing.assert_close(
            td["features"],
            torch.tensor(
                [
                    [1, 2, 3, 4, 5],
                    [2, 3, 4, 5, 6],
                ],
                dtype=torch.float32,
                device=self.device,
            ),
        )
        torch.testing.assert_close(
            td["label"],
            torch.tensor(
                [
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                ],
                dtype=torch.int64,
                device=self.device,
            ),
        )
        # the episode that ended (n.2) should not have -1 as predicted class
        assert td["predicted_class"][1].item() != -1, (
            "Expected a non -1 predicted class"
        )
        # first episode should still have -1 as predicted class
        assert td["predicted_class"][0].item() == -1, "Expected a -1 predicted class"
        torch.testing.assert_close(
            td["done"],
            torch.tensor(
                [
                    [0],
                    [1],
                ],
                dtype=torch.bool,
                device=self.device,
            ),
        )

    def test_state3_with_reset(self):
        td = self.get_state3_with_reset()
        assert td.batch_size == torch.Size((2,))
        torch.testing.assert_close(
            td["action_mask"],
            torch.tensor(
                [
                    [1, 0, 1, 0, 1, 0],
                    [1, 1, 1, 1, 1, 1],
                ],
                dtype=torch.bool,
                device=self.device,
            ),
        )
        torch.testing.assert_close(
            td["feature_mask"],
            torch.tensor(
                [
                    [1, 0, 1, 0, 1],
                    [0, 0, 0, 0, 0],
                ],
                dtype=torch.bool,
                device=self.device,
            ),
        )
        torch.testing.assert_close(
            td["masked_features"],
            torch.tensor(
                [
                    [1, 0, 3, 0, 5],
                    [0, 0, 0, 0, 0],
                ],
                dtype=torch.float32,
                device=self.device,
            ),
        )
        torch.testing.assert_close(
            td["fa_reward"],
            torch.tensor(
                [
                    [-14],
                    [0],
                ],
                dtype=torch.float32,
                device=self.device,
            ),
        )
        # only check that the first sample does not have a model reward
        torch.testing.assert_close(
            td["model_reward"][0],
            torch.tensor(
                [0],
                dtype=torch.float32,
                device=self.device,
            ),
        )
        torch.testing.assert_close(
            td["invalid_action_reward"],
            torch.tensor(
                [
                    [0],
                    [0],
                ],
                dtype=torch.float32,
                device=self.device,
            ),
        )
        # NOTE: this is a bit weird, but as soon as a single batch element is done, a complete new data batch
        # (from dataset_fn) is loaded, and only some of the elements are used. This is why the all_features tensor skips over the sample [3, 4, 5, 6, 7]
        torch.testing.assert_close(
            td["features"],
            torch.tensor(
                [
                    [1, 2, 3, 4, 5],
                    [4, 5, 6, 7, 8],
                ],
                dtype=torch.float32,
                device=self.device,
            ),
        )
        torch.testing.assert_close(
            td["predicted_class"],
            torch.tensor(
                [
                    [-1],
                    [-1],
                ],
                dtype=torch.int64,
                device=self.device,
            ),
        )
        torch.testing.assert_close(
            td["label"],
            torch.tensor(
                [
                    [1, 0, 0, 0],
                    [0, 0, 0, 1],
                ],
                dtype=self.td["label"].dtype,
                device=self.device,
            ),
        )
        torch.testing.assert_close(
            td["done"],
            torch.tensor(
                [
                    [0],
                    [0],
                ],
                dtype=torch.bool,
                device=self.device,
            ),
        )
