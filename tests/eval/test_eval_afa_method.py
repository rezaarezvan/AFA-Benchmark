"""
Checks whether eval_afa_method works with DummyAFAMethod and CubeDataset.
"""

import argparse
from eval.scripts.eval_afa_method import eval_afa_method


def test_eval_afa_method_sequential_dummy():
    args: argparse.Namespace = argparse.Namespace(
        afa_method_name="sequential_dummy",
        afa_method_path="models/afa_rl/sequential_dummy-cube_train_10000.pt",
        dataset_name="cube",
        dataset_path="data/cube/cube_val_1000.pt",
    )

    eval_afa_method(args)


def test_eval_afa_method_random_dummy():
    args: argparse.Namespace = argparse.Namespace(
        afa_method_name="random_dummy",
        afa_method_path="models/afa_rl/random_dummy-cube_train_10000.pt",
        dataset_name="cube",
        dataset_path="data/cube/cube_val_1000.pt",
    )

    eval_afa_method(args)
