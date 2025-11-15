import torch
import pytest

from afabench.afa_rl.afa_env import AFAEnv
from afabench.common.datasets import CubeDataset
from afabench.common.warm_start import (
    FixedRandomStrategy,
    LeastInformativeStrategy,
    MutualInformationStrategy,
    RandomPerEpisodeStrategy,
)


@pytest.fixture
def cube_dataset():
    """Generate CUBE dataset locally."""
    dataset = CubeDataset(n_samples=100, seed=42)
    dataset.generate_data()
    return dataset


@pytest.fixture
def dataset_fn_factory():
    """Factory to create dataset_fn like in training scripts."""

    def make_dataset_fn(features, labels):
        def dataset_fn(batch_size):
            idx = torch.randint(0, len(features), (batch_size[0],))
            return features[idx], labels[idx]

        return dataset_fn

    return make_dataset_fn


@pytest.fixture
def simple_reward_fn():
    """Simple reward function."""

    def reward_fn(td):
        n_features = td["feature_mask"].sum(dim=-1, keepdim=True)
        return -n_features.float()

    return reward_fn


class TestWarmStartWithCUBE:
    """Test warm-start with CUBE dataset."""

    def test_fixed_strategy_with_cube(self, cube_dataset):
        """Test FixedRandomStrategy with CUBE."""
        strategy = FixedRandomStrategy(seed=42)

        selected = strategy.select_features(
            n_features_total=cube_dataset.n_features,
            n_features_select=2,
            train_features=cube_dataset.features,
            train_labels=cube_dataset.labels.argmax(dim=-1),
        )

        assert len(selected) == 2
        assert len(set(selected)) == 2
        assert all(0 <= idx < cube_dataset.n_features for idx in selected)

        # Verify consistency
        selected2 = strategy.select_features(
            n_features_total=cube_dataset.n_features,
            n_features_select=2,
        )
        assert selected == selected2

    def test_mi_strategy_with_cube(self, cube_dataset):
        """Test MutualInformationStrategy with CUBE."""
        strategy = MutualInformationStrategy(seed=42)

        selected = strategy.select_features(
            n_features_total=cube_dataset.n_features,
            n_features_select=3,
            train_features=cube_dataset.features,
            train_labels=cube_dataset.labels.argmax(dim=-1),
        )

        assert len(selected) == 3
        # CUBE: features 0-9 informative, 10-19 dummy
        informative_selected = sum(1 for idx in selected if idx < 10)
        assert informative_selected >= 2

    def test_least_informative_with_cube(self, cube_dataset):
        """Test LeastInformativeStrategy with CUBE."""
        strategy = LeastInformativeStrategy(seed=42)

        selected = strategy.select_features(
            n_features_total=cube_dataset.n_features,
            n_features_select=3,
            train_features=cube_dataset.features,
            train_labels=cube_dataset.labels.argmax(dim=-1),
        )

        assert len(selected) == 3
        # Should select mostly dummy features (10-19)
        dummy_selected = sum(1 for idx in selected if idx >= 10)
        assert dummy_selected >= 2

    def test_random_per_episode_with_cube(self, cube_dataset):
        """Test RandomPerEpisodeStrategy with CUBE."""
        strategy = RandomPerEpisodeStrategy(seed=42)

        selections = [
            strategy.select_features(
                n_features_total=cube_dataset.n_features,
                n_features_select=2,
            )
            for _ in range(5)
        ]

        for sel in selections:
            assert len(sel) == 2
            assert all(0 <= idx < cube_dataset.n_features for idx in sel)

        unique_selections = set(tuple(sorted(s)) for s in selections)
        assert len(unique_selections) >= 2


class TestAFAEnvIntegration:
    """Test warm-start workflow with AFAEnv."""

    def test_afaenv_without_warmstart(
        self, cube_dataset, dataset_fn_factory, simple_reward_fn
    ):
        """Baseline: AFAEnv works without warm-start."""
        dataset_fn = dataset_fn_factory(
            cube_dataset.features, cube_dataset.labels
        )

        env = AFAEnv(
            dataset_fn=dataset_fn,
            reward_fn=simple_reward_fn,
            device=torch.device("cpu"),
            batch_size=torch.Size((2,)),
            feature_size=cube_dataset.n_features,
            n_classes=cube_dataset.n_classes,
            hard_budget=5,
        )

        td = env.reset()
        assert td["feature_mask"].sum() == 0

    def test_warm_start_manual_application(
        self, cube_dataset, dataset_fn_factory, simple_reward_fn
    ):
        """Test manually applying warm-start to feature masks."""
        strategy = FixedRandomStrategy(seed=42)
        selected = strategy.select_features(
            n_features_total=cube_dataset.n_features,
            n_features_select=2,
        )

        batch_size = 3
        feature_mask = torch.zeros(
            batch_size, cube_dataset.n_features, dtype=torch.bool
        )
        for idx in selected:
            feature_mask[:, idx] = True

        assert feature_mask.sum() == batch_size * 2
        for idx in selected:
            assert feature_mask[:, idx].all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
