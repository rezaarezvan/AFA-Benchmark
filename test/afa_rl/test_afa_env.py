import torch

from afabench.afa_rl.afa_env import AFAEnv
from afabench.afa_rl.datasets import get_afa_dataset_fn
from afabench.afa_rl.reward_functions import get_fixed_reward_reward_fn
from afabench.common.initializers.fixed_random_initializer import (
    FixedRandomInitializer,
)
from afabench.common.unmaskers import ImagePatchUnmasker
from afabench.common.unmaskers.direct_unmasker import DirectUnmasker


def test_initializer_and_unmasker_integration() -> None:
    # Create 2-channel images with simple patterns for easy testing
    # Each sample has 2 channels, each channel is 4x4
    all_features = torch.tensor(
        [
            # Sample 1: channel 0 has sequential values 1-16, channel 1 has values 17-32
            [
                [
                    [1.0, 2.0, 3.0, 4.0],
                    [5.0, 6.0, 7.0, 8.0],
                    [9.0, 10.0, 11.0, 12.0],
                    [13.0, 14.0, 15.0, 16.0],
                ],  # Channel 0
                [
                    [17.0, 18.0, 19.0, 20.0],
                    [21.0, 22.0, 23.0, 24.0],
                    [25.0, 26.0, 27.0, 28.0],
                    [29.0, 30.0, 31.0, 32.0],
                ],  # Channel 1
            ],
            # Sample 2: channel 0 has even values 2-32, channel 1 has odd values 1-31
            [
                [
                    [2.0, 4.0, 6.0, 8.0],
                    [10.0, 12.0, 14.0, 16.0],
                    [18.0, 20.0, 22.0, 24.0],
                    [26.0, 28.0, 30.0, 32.0],
                ],  # Channel 0
                [
                    [1.0, 3.0, 5.0, 7.0],
                    [9.0, 11.0, 13.0, 15.0],
                    [17.0, 19.0, 21.0, 23.0],
                    [25.0, 27.0, 29.0, 31.0],
                ],  # Channel 1
            ],
            # Sample 3: channel 0 has values 0-15, channel 1 has values 16-31
            [
                [
                    [0.0, 1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0, 7.0],
                    [8.0, 9.0, 10.0, 11.0],
                    [12.0, 13.0, 14.0, 15.0],
                ],  # Channel 0
                [
                    [16.0, 17.0, 18.0, 19.0],
                    [20.0, 21.0, 22.0, 23.0],
                    [24.0, 25.0, 26.0, 27.0],
                    [28.0, 29.0, 30.0, 31.0],
                ],  # Channel 1
            ],
        ]
    )
    all_labels = torch.tensor(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ]
    )
    dataset_fn = get_afa_dataset_fn(all_features, all_labels)
    env = AFAEnv(
        dataset_fn=dataset_fn,
        reward_fn=get_fixed_reward_reward_fn(
            reward_for_stop=0.0, reward_otherwise=-1.0
        ),
        device=torch.device("cpu"),
        batch_size=torch.Size((2,)),
        feature_shape=torch.Size((2, 4, 4)),
        n_selections=4,
        n_classes=3,
        hard_budget=2,
        initialize_fn=FixedRandomInitializer(unmask_ratio=0.0).initialize,
        unmask_fn=ImagePatchUnmasker(
            image_side_length=4,
            patch_size=2,
            n_channels=2,
        ).unmask,
        seed=123,
    )

    # Get initial state, t = 0
    td = env.reset()
    assert torch.allclose(
        td["feature_mask"], torch.zeros((2, 2, 4, 4), dtype=torch.bool)
    )
    assert torch.allclose(
        td["performed_action_mask"], torch.zeros((2, 5), dtype=torch.bool)
    )
    assert torch.allclose(
        td["allowed_action_mask"], torch.ones((2, 5), dtype=torch.bool)
    )
    assert torch.allclose(
        td["performed_selection_mask"], torch.zeros((2, 4), dtype=torch.bool)
    )
    assert torch.allclose(
        td["masked_features"], torch.zeros((2, 2, 4, 4), dtype=torch.float32)
    )
    # Trust that features and labels are forwarded properly by the environment

    # Pick the second patch for the first sample and the third patch for the second sample
    td["action"] = torch.tensor([[2], [3]], dtype=torch.int64)

    # t = 1
    td = env.step(td)
    td = td["next"]

    expected_feature_mask_t1 = torch.tensor(
        [
            # Sample 1: second patch unmasked (patch 2: top-right 2x2)
            [
                [
                    [False, False, True, True],
                    [False, False, True, True],
                    [False, False, False, False],
                    [False, False, False, False],
                ],
                [
                    [False, False, True, True],
                    [False, False, True, True],
                    [False, False, False, False],
                    [False, False, False, False],
                ],
            ],
            # Sample 2: third patch unmasked (patch 3: bottom-left 2x2)
            [
                [
                    [False, False, False, False],
                    [False, False, False, False],
                    [True, True, False, False],
                    [True, True, False, False],
                ],
                [
                    [False, False, False, False],
                    [False, False, False, False],
                    [True, True, False, False],
                    [True, True, False, False],
                ],
            ],
        ],
        dtype=torch.bool,
    )
    assert torch.allclose(td["feature_mask"], expected_feature_mask_t1)

    # Check masks - actions 2 and 3 should now be performed and disabled
    expected_performed_action_mask_t1 = torch.tensor(
        [
            [False, False, True, False, False],  # Sample 1: action 2 performed
            [False, False, False, True, False],  # Sample 2: action 3 performed
        ],
        dtype=torch.bool,
    )
    expected_allowed_action_mask_t1 = torch.tensor(
        [
            [True, True, False, True, True],  # Sample 1: action 2 disabled
            [True, True, True, False, True],  # Sample 2: action 3 disabled
        ],
        dtype=torch.bool,
    )
    expected_performed_selection_mask_t1 = torch.tensor(
        [
            [False, True, False, False],  # Sample 1: selection 1 performed
            [False, False, True, False],  # Sample 2: selection 2 performed
        ],
        dtype=torch.bool,
    )
    assert torch.allclose(
        td["performed_action_mask"], expected_performed_action_mask_t1
    )
    assert torch.allclose(
        td["allowed_action_mask"], expected_allowed_action_mask_t1
    )
    assert torch.allclose(
        td["performed_selection_mask"], expected_performed_selection_mask_t1
    )

    # Check masked_features - should show the unmasked values
    expected_masked_features_t1 = torch.zeros(
        (2, 2, 4, 4), dtype=torch.float32
    )
    # Sample 1: patch 2 (top-right) unmasked
    expected_masked_features_t1[0, 0, 0:2, 2:4] = torch.tensor(
        [[3.0, 4.0], [7.0, 8.0]]
    )
    expected_masked_features_t1[0, 1, 0:2, 2:4] = torch.tensor(
        [[19.0, 20.0], [23.0, 24.0]]
    )
    # Sample 2: patch 3 (bottom-left) unmasked
    expected_masked_features_t1[1, 0, 2:4, 0:2] = torch.tensor(
        [[18.0, 20.0], [26.0, 28.0]]
    )
    expected_masked_features_t1[1, 1, 2:4, 0:2] = torch.tensor(
        [[17.0, 19.0], [25.0, 27.0]]
    )
    assert torch.allclose(td["masked_features"], expected_masked_features_t1)

    # Pick the first patch for the first sample and the fourth patch for the second sample
    td["action"] = torch.tensor([[1], [4]], dtype=torch.int64)

    # t = 2 (final step due to hard_budget=2)
    td = env.step(td)
    td = td["next"]
    expected_feature_mask_t2 = torch.tensor(
        [
            # Sample 1: patches 1 and 2 unmasked
            [
                [
                    [True, True, True, True],
                    [True, True, True, True],
                    [False, False, False, False],
                    [False, False, False, False],
                ],
                [
                    [True, True, True, True],
                    [True, True, True, True],
                    [False, False, False, False],
                    [False, False, False, False],
                ],
            ],
            # Sample 2: patches 3 and 4 unmasked
            [
                [
                    [False, False, False, False],
                    [False, False, False, False],
                    [True, True, True, True],
                    [True, True, True, True],
                ],
                [
                    [False, False, False, False],
                    [False, False, False, False],
                    [True, True, True, True],
                    [True, True, True, True],
                ],
            ],
        ],
        dtype=torch.bool,
    )
    assert torch.allclose(td["feature_mask"], expected_feature_mask_t2)

    # Check masked_features - should show both patches unmasked
    expected_masked_features_t2 = torch.zeros(
        (2, 2, 4, 4), dtype=torch.float32
    )
    # Sample 1: patches 1 (top-left) and 2 (top-right) unmasked
    expected_masked_features_t2[0, 0, 0:2, 0:2] = torch.tensor(
        [[1.0, 2.0], [5.0, 6.0]]
    )
    expected_masked_features_t2[0, 0, 0:2, 2:4] = torch.tensor(
        [[3.0, 4.0], [7.0, 8.0]]
    )
    expected_masked_features_t2[0, 1, 0:2, 0:2] = torch.tensor(
        [[17.0, 18.0], [21.0, 22.0]]
    )
    expected_masked_features_t2[0, 1, 0:2, 2:4] = torch.tensor(
        [[19.0, 20.0], [23.0, 24.0]]
    )
    # Sample 2: patches 3 (bottom-left) and 4 (bottom-right) unmasked
    expected_masked_features_t2[1, 0, 2:4, 0:2] = torch.tensor(
        [[18.0, 20.0], [26.0, 28.0]]
    )
    expected_masked_features_t2[1, 0, 2:4, 2:4] = torch.tensor(
        [[22.0, 24.0], [30.0, 32.0]]
    )
    expected_masked_features_t2[1, 1, 2:4, 0:2] = torch.tensor(
        [[17.0, 19.0], [25.0, 27.0]]
    )
    expected_masked_features_t2[1, 1, 2:4, 2:4] = torch.tensor(
        [[21.0, 23.0], [29.0, 31.0]]
    )
    assert torch.allclose(td["masked_features"], expected_masked_features_t2)

    # Episode should be done due to hard budget
    assert td["done"].all(), "Episode should be terminated due to hard budget"


def test_stop_due_to_hard_budget() -> None:
    """Test that the environment terminates when hard budget is reached."""
    # Use simple 1D features for easy testing
    all_features = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],  # Sample 1
            [5.0, 6.0, 7.0, 8.0],  # Sample 2
        ]
    )
    all_labels = torch.tensor(
        [
            [1, 0],  # Sample 1 label
            [0, 1],  # Sample 2 label
        ]
    )
    dataset_fn = get_afa_dataset_fn(all_features, all_labels, shuffle=False)

    # Create environment with hard budget of 2
    env = AFAEnv(
        dataset_fn=dataset_fn,
        reward_fn=get_fixed_reward_reward_fn(
            reward_for_stop=0.0, reward_otherwise=-1.0
        ),
        device=torch.device("cpu"),
        batch_size=torch.Size((2,)),
        feature_shape=torch.Size((4,)),
        n_selections=4,  # 4 possible selections
        n_classes=2,
        hard_budget=2,  # Should terminate after 2 selections
        initialize_fn=FixedRandomInitializer(unmask_ratio=0.0).initialize,
        unmask_fn=DirectUnmasker().unmask,
        seed=123,
    )

    # Reset environment
    td = env.reset()

    # First selection - should not terminate
    td["action"] = torch.tensor(
        [[1], [2]], dtype=torch.int64
    )  # Select features 1 and 2
    td = env.step(td)
    td = td["next"]

    assert not td["done"].any(), (
        "Environment should not terminate after first selection"
    )

    # Second selection - should terminate due to hard budget
    td["action"] = torch.tensor(
        [[3], [4]], dtype=torch.int64
    )  # Select features 3 and 4
    td = env.step(td)
    td = td["next"]

    assert td["done"].all(), (
        "Environment should terminate after reaching hard budget of 2"
    )


def test_stop_due_to_no_more_actions() -> None:
    """Test that the environment terminates when all selection actions are exhausted."""
    # Use simple 1D features with only 2 features available for selection
    all_features = torch.tensor(
        [
            [1.0, 2.0],  # Sample 1 - only 2 features
            [3.0, 4.0],  # Sample 2 - only 2 features
        ]
    )
    all_labels = torch.tensor(
        [
            [1, 0],  # Sample 1 label
            [0, 1],  # Sample 2 label
        ]
    )
    dataset_fn = get_afa_dataset_fn(all_features, all_labels, shuffle=False)

    # Create environment with only 2 selections possible and high hard budget
    env = AFAEnv(
        dataset_fn=dataset_fn,
        reward_fn=get_fixed_reward_reward_fn(
            reward_for_stop=0.0, reward_otherwise=-1.0
        ),
        device=torch.device("cpu"),
        batch_size=torch.Size((2,)),
        feature_shape=torch.Size((2,)),
        n_selections=2,  # Only 2 possible selections
        n_classes=2,
        hard_budget=10,  # High budget, should terminate due to no more actions
        initialize_fn=FixedRandomInitializer(unmask_ratio=0.0).initialize,
        unmask_fn=DirectUnmasker().unmask,
        seed=123,
    )

    # Reset environment
    td = env.reset()

    # First selection - should not terminate
    td["action"] = torch.tensor(
        [[1], [1]], dtype=torch.int64
    )  # Select first feature
    td = env.step(td)
    td = td["next"]

    assert not td["done"].any(), (
        "Environment should not terminate after first selection"
    )

    # Second selection - should terminate because no more selection actions available
    td["action"] = torch.tensor(
        [[2], [2]], dtype=torch.int64
    )  # Select second feature
    td = env.step(td)
    td = td["next"]

    assert td["done"].all(), (
        "Environment should auto-terminate when no more selection actions are available"
    )


def test_stop_due_to_stop_action() -> None:
    """Test that the environment terminates when stop action (action 0) is chosen."""
    # Use simple 1D features
    all_features = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],  # Sample 1
            [5.0, 6.0, 7.0, 8.0],  # Sample 2
        ]
    )
    all_labels = torch.tensor(
        [
            [1, 0],  # Sample 1 label
            [0, 1],  # Sample 2 label
        ]
    )
    dataset_fn = get_afa_dataset_fn(all_features, all_labels, shuffle=False)

    # Create environment with high hard budget - should terminate due to stop action
    env = AFAEnv(
        dataset_fn=dataset_fn,
        reward_fn=get_fixed_reward_reward_fn(
            reward_for_stop=0.0, reward_otherwise=-1.0
        ),
        device=torch.device("cpu"),
        batch_size=torch.Size((2,)),
        feature_shape=torch.Size((4,)),
        n_selections=4,
        n_classes=2,
        hard_budget=10,  # High budget, should not terminate due to budget
        initialize_fn=FixedRandomInitializer(unmask_ratio=0.0).initialize,
        unmask_fn=DirectUnmasker().unmask,
        seed=123,
    )

    # Reset environment
    td = env.reset()

    # Make one selection first - should not terminate
    td["action"] = torch.tensor(
        [[1], [2]], dtype=torch.int64
    )  # Select features 1 and 2
    td = env.step(td)
    td = td["next"]

    assert not td["done"].any(), (
        "Environment should not terminate after first selection"
    )

    # Choose stop action (action 0) - should terminate
    td["action"] = torch.tensor([[0], [0]], dtype=torch.int64)  # Stop action
    td = env.step(td)
    td = td["next"]

    assert td["done"].all(), (
        "Environment should terminate when stop action is chosen"
    )


def test_per_sample_termination_hard_budget() -> None:
    """Test that samples terminate independently when reaching hard budget."""
    # Use simple 1D features
    all_features = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],  # Sample 1
            [5.0, 6.0, 7.0, 8.0],  # Sample 2
            [9.0, 10.0, 11.0, 12.0],  # Sample 3
        ]
    )
    all_labels = torch.tensor(
        [
            [1, 0, 0],  # Sample 1 label
            [0, 1, 0],  # Sample 2 label
            [0, 0, 1],  # Sample 3 label
        ]
    )
    dataset_fn = get_afa_dataset_fn(all_features, all_labels, shuffle=False)

    # Create environment with different hard budgets for samples
    # We'll simulate this by carefully choosing actions
    env = AFAEnv(
        dataset_fn=dataset_fn,
        reward_fn=get_fixed_reward_reward_fn(
            reward_for_stop=0.0, reward_otherwise=-1.0
        ),
        device=torch.device("cpu"),
        batch_size=torch.Size((3,)),
        feature_shape=torch.Size((4,)),
        n_selections=4,
        n_classes=3,
        hard_budget=2,  # All samples have same hard budget
        initialize_fn=FixedRandomInitializer(unmask_ratio=0.0).initialize,
        unmask_fn=DirectUnmasker().unmask,
        seed=123,
    )

    # Reset environment
    td = env.reset()

    # First step: Sample 1 makes selection, samples 2&3 choose stop
    td["action"] = torch.tensor([[1], [0], [0]], dtype=torch.int64)
    td = env.step(td)
    td = td["next"]

    # Only samples 2 and 3 should be done (they chose stop)
    expected_done = torch.tensor([[False], [True], [True]], dtype=torch.bool)
    assert torch.equal(td["done"], expected_done), (
        f"Expected done={expected_done}, got {td['done']}"
    )

    # Second step: Sample 1 makes another selection (reaches hard budget)
    td["action"] = torch.tensor([[2], [0], [0]], dtype=torch.int64)
    td = env.step(td)
    td = td["next"]

    # Now all samples should be done
    assert td["done"].all(), (
        "All samples should be done - sample 1 reached hard budget, others already stopped"
    )


def test_per_sample_termination_no_more_actions() -> None:
    """Test that samples terminate independently when exhausting actions."""
    # Use simple 1D features with only 2 features available for selection
    all_features = torch.tensor(
        [
            [1.0, 2.0],  # Sample 1
            [3.0, 4.0],  # Sample 2
            [5.0, 6.0],  # Sample 3
        ]
    )
    all_labels = torch.tensor(
        [
            [1, 0, 0],  # Sample 1 label
            [0, 1, 0],  # Sample 2 label
            [0, 0, 1],  # Sample 3 label
        ]
    )
    dataset_fn = get_afa_dataset_fn(all_features, all_labels, shuffle=False)

    # Create environment with only 2 selection actions possible
    # We'll simulate different exhaustion times by having samples stop at different points
    env = AFAEnv(
        dataset_fn=dataset_fn,
        reward_fn=get_fixed_reward_reward_fn(
            reward_for_stop=0.0, reward_otherwise=-1.0
        ),
        device=torch.device("cpu"),
        batch_size=torch.Size((3,)),
        feature_shape=torch.Size((2,)),
        n_selections=2,
        n_classes=3,
        hard_budget=10,  # High budget
        initialize_fn=FixedRandomInitializer(unmask_ratio=0.0).initialize,
        unmask_fn=DirectUnmasker().unmask,
        seed=123,
    )

    # Reset environment
    td = env.reset()

    # Step 1: Sample 1 makes first selection, sample 2 exhausts all selections, sample 3 makes first selection
    td["action"] = torch.tensor([[1], [1], [1]], dtype=torch.int64)
    td = env.step(td)
    td = td["next"]

    # No samples should be done yet
    expected_done = torch.tensor([[False], [False], [False]], dtype=torch.bool)
    assert torch.equal(td["done"], expected_done), (
        f"Expected done={expected_done}, got {td['done']}"
    )

    # Step 2: Sample 1 stops, sample 2 makes second selection (should auto-terminate), sample 3 makes second selection (should auto-terminate)
    td["action"] = torch.tensor([[0], [2], [2]], dtype=torch.int64)
    td = env.step(td)
    td = td["next"]

    # Sample 1 should be done (chose stop), samples 2 and 3 should be done (exhausted all selections)
    expected_done = torch.tensor([[True], [True], [True]], dtype=torch.bool)
    assert torch.equal(td["done"], expected_done), (
        f"Expected done={expected_done}, got {td['done']}"
    )


def test_per_sample_termination_stop_action() -> None:
    """Test that samples terminate independently when choosing stop action."""
    # Use simple 1D features
    all_features = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],  # Sample 1
            [5.0, 6.0, 7.0, 8.0],  # Sample 2
            [9.0, 10.0, 11.0, 12.0],  # Sample 3
        ]
    )
    all_labels = torch.tensor(
        [
            [1, 0, 0],  # Sample 1 label
            [0, 1, 0],  # Sample 2 label
            [0, 0, 1],  # Sample 3 label
        ]
    )
    dataset_fn = get_afa_dataset_fn(all_features, all_labels, shuffle=False)

    env = AFAEnv(
        dataset_fn=dataset_fn,
        reward_fn=get_fixed_reward_reward_fn(
            reward_for_stop=0.0, reward_otherwise=-1.0
        ),
        device=torch.device("cpu"),
        batch_size=torch.Size((3,)),
        feature_shape=torch.Size((4,)),
        n_selections=4,
        n_classes=3,
        hard_budget=10,  # High budget
        initialize_fn=FixedRandomInitializer(unmask_ratio=0.0).initialize,
        unmask_fn=DirectUnmasker().unmask,
        seed=123,
    )

    # Reset environment
    td = env.reset()

    # Step 1: Sample 1 chooses stop, others make selections
    td["action"] = torch.tensor([[0], [1], [2]], dtype=torch.int64)
    td = env.step(td)
    td = td["next"]

    # Only sample 1 should be done
    expected_done = torch.tensor([[True], [False], [False]], dtype=torch.bool)
    assert torch.equal(td["done"], expected_done), (
        f"Expected done={expected_done}, got {td['done']}"
    )

    # Step 2: All remaining samples make selections
    td["action"] = torch.tensor([[0], [3], [1]], dtype=torch.int64)
    td = env.step(td)
    td = td["next"]

    # Still only sample 1 should be done
    expected_done = torch.tensor([[True], [False], [False]], dtype=torch.bool)
    assert torch.equal(td["done"], expected_done), (
        f"Expected done={expected_done}, got {td['done']}"
    )

    # Step 3: Sample 2 chooses stop, sample 3 makes selection
    td["action"] = torch.tensor([[0], [0], [4]], dtype=torch.int64)
    td = env.step(td)
    td = td["next"]

    # Samples 1 and 2 should be done
    expected_done = torch.tensor([[True], [True], [False]], dtype=torch.bool)
    assert torch.equal(td["done"], expected_done), (
        f"Expected done={expected_done}, got {td['done']}"
    )

    # Step 4: Sample 3 chooses stop
    td["action"] = torch.tensor([[0], [0], [0]], dtype=torch.int64)
    td = env.step(td)
    td = td["next"]

    # All samples should be done
    assert td["done"].all(), "All samples should be done"
