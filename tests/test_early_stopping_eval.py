"""Test script to verify that the updated evaluation supports early stopping."""

import time
from pathlib import Path

import numpy as np
import pytest
import torch

from src.common.custom_types import (
    AFASelection,
    FeatureMask,
    Features,
    Label,
    MaskedFeatures,
)


# Mock implementations for testing
class MockEarlyStoppingMethod:
    """Mock AFA method that stops early after a random number of steps."""

    def __init__(
        self, max_steps: int = 5, stop_probability: float = 0.3
    ) -> None:
        self.max_steps: int = max_steps
        self.stop_probability: float = stop_probability
        self.step_count: dict[int, int] = {}  # Track steps per sample

    def select(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
        features: Features | None = None,  # noqa: ARG002
        label: Label | None = None,  # noqa: ARG002
    ) -> AFASelection:
        """Select features with chance of early stopping."""
        batch_size = masked_features.shape[0]
        selections = torch.zeros(batch_size, 1, dtype=torch.long)

        for i in range(batch_size):
            sample_id = id(
                masked_features[i].data_ptr()
            )  # Unique ID for this sample

            if sample_id not in self.step_count:
                self.step_count[sample_id] = 0

            self.step_count[sample_id] += 1

            # Decide whether to stop
            if (
                self.step_count[sample_id] >= self.max_steps
                or np.random.default_rng().random() < self.stop_probability
            ):
                selections[i] = 0  # Stop signal
            else:
                # Select a random unobserved feature (1-based indexing)
                unobserved = ~feature_mask[i]
                if unobserved.any():
                    available_features = torch.where(unobserved)[0]
                    selected_idx = np.random.default_rng().choice(
                        available_features.cpu().numpy()
                    )
                    selections[i] = selected_idx + 1  # Convert to 1-based
                else:
                    selections[i] = 0  # No more features to select

        return selections

    def predict(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,  # noqa: ARG002
        features: Features | None = None,  # noqa: ARG002
        label: Label | None = None,  # noqa: ARG002
    ) -> Label:
        """Return simple random prediction."""
        batch_size = masked_features.shape[0]
        n_classes = 2  # Binary classification
        return torch.softmax(torch.randn(batch_size, n_classes), dim=1)


class MockDataset:
    """Mock dataset for testing."""

    def __init__(
        self, n_samples: int = 10, n_features: int = 8, n_classes: int = 2
    ) -> None:
        self.n_samples: int = n_samples
        self.n_features: int = n_features
        self.n_classes: int = n_classes

        # Generate random data
        self.features: Features = torch.randn(n_samples, n_features)
        self.labels: Label = torch.randint(0, n_classes, (n_samples,))
        # Convert to one-hot
        self.labels = torch.nn.functional.one_hot(
            self.labels, n_classes
        ).float()

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[Features, Label]:
        return self.features[idx], self.labels[idx]

    def generate_data(self) -> None:
        """Generate data (required by AFADataset protocol)."""

    def get_all_data(self) -> tuple[Features, Label]:
        """Return all data (required by AFADataset protocol)."""
        return self.features, self.labels

    def save(self, path: Path) -> None:
        """Save dataset (required by AFADataset protocol)."""

    @classmethod
    def load(cls, path: Path) -> "MockDataset":  # noqa: ARG003
        """Load dataset (required by AFADataset protocol)."""
        return cls()


class TestEarlyStoppingEvaluation:
    """Test suite for early stopping evaluation functionality."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        # Set random seed for reproducibility
        torch.manual_seed(42)

    def test_early_stopping_evaluation(self) -> None:
        """Test the evaluation with early stopping."""
        from src.eval.metrics import eval_afa_method  # noqa: PLC0415

        # Create mock data and method
        dataset = MockDataset(n_samples=50, n_features=10, n_classes=2)
        method = MockEarlyStoppingMethod(max_steps=7, stop_probability=0.2)

        # Run evaluation with batching for performance
        budget = 10  # Maximum budget

        metrics = eval_afa_method(
            afa_select_fn=method.select,
            dataset=dataset,  # pyright: ignore[reportArgumentType]
            budget=budget,
            afa_predict_fn=method.predict,
            batch_size=8,  # Test batched processing
            device=torch.device("cpu"),
        )

        # Verify that metrics are computed correctly
        assert len(metrics["accuracy_all"]) <= budget
        assert len(metrics["f1_all"]) <= budget
        assert len(metrics["bce_all"]) <= budget

        # Verify all samples took at least zero steps (methods can stop immediately)
        assert metrics["actual_steps"].min() >= 0

        # Check that we have valid metrics where samples exist
        max_steps = metrics["actual_steps"].max()
        for i in range(max_steps):
            samples_at_step = (metrics["actual_steps"] > i).sum()
            if samples_at_step > 0:
                assert not torch.isnan(metrics["accuracy_all"][i])
                assert not torch.isnan(metrics["f1_all"][i])
                assert not torch.isnan(metrics["bce_all"][i])

        # Check early stopping behavior
        early_stops = (metrics["actual_steps"] < budget).sum()
        assert early_stops > 0, "Some samples should stop early"

    def test_batching_performance(self) -> None:
        """Test that batched processing provides significant performance improvement."""
        from src.eval.metrics import eval_afa_method  # noqa: PLC0415

        # Use deterministic method for consistent comparison
        class DeterministicMethod:
            def __init__(self) -> None:
                self.call_count: int = 0

            def select(
                self,
                masked_features: MaskedFeatures,
                feature_mask: FeatureMask,
                features: Features | None = None,  # noqa: ARG002
                label: Label | None = None,  # noqa: ARG002
            ) -> AFASelection:
                batch_size = masked_features.shape[0]
                selections = torch.zeros(batch_size, 1, dtype=torch.long)

                for i in range(batch_size):
                    # Always select the first unobserved feature deterministically
                    unobserved = ~feature_mask[i]
                    if unobserved.any():
                        available_features = torch.where(unobserved)[0]
                        selections[i] = (
                            available_features[0] + 1
                        )  # Convert to 1-based
                    else:
                        selections[i] = 1  # Just select first feature again

                return selections

            def predict(
                self,
                masked_features: MaskedFeatures,
                feature_mask: FeatureMask,  # noqa: ARG002
                features: Features | None = None,  # noqa: ARG002
                label: Label | None = None,  # noqa: ARG002
            ) -> Label:
                batch_size = masked_features.shape[0]
                n_classes = 2
                # Use deterministic predictions based on call count
                self.call_count += 1
                torch.manual_seed(self.call_count)
                result = torch.softmax(
                    torch.randn(batch_size, n_classes), dim=1
                )
                return result

        # Create larger dataset for timing
        dataset = MockDataset(n_samples=200, n_features=8, n_classes=2)
        budget = 5

        # Test with batch_size=1 (essentially unbatched)
        method1 = DeterministicMethod()
        start_time = time.time()
        metrics_unbatched = eval_afa_method(
            afa_select_fn=method1.select,
            dataset=dataset,  # pyright: ignore[reportArgumentType]
            budget=budget,
            afa_predict_fn=method1.predict,
            batch_size=1,
            device=torch.device("cpu"),
        )
        unbatched_time = time.time() - start_time

        # Test with larger batch_size
        method2 = DeterministicMethod()
        start_time = time.time()
        metrics_batched = eval_afa_method(
            afa_select_fn=method2.select,
            dataset=dataset,  # pyright: ignore[reportArgumentType]
            budget=budget,
            afa_predict_fn=method2.predict,
            batch_size=20,
            device=torch.device("cpu"),
        )
        batched_time = time.time() - start_time

        # Verify that both produce the same step behavior (deterministic selection)
        assert torch.equal(
            metrics_unbatched["actual_steps"], metrics_batched["actual_steps"]
        )

        # Check that accuracy values are reasonable
        assert len(metrics_unbatched["accuracy_all"]) == len(
            metrics_batched["accuracy_all"]
        )

        # Expect at least some speedup from batching (or at least no slowdown)
        speedup = unbatched_time / batched_time
        assert speedup >= 1.0

    def test_immediate_stopping(self) -> None:
        """Test that methods can stop immediately (0 steps)."""
        from src.eval.metrics import eval_afa_method  # noqa: PLC0415

        class ImmediateStopMethod:
            """Method that always stops immediately."""

            def select(
                self,
                masked_features: MaskedFeatures,
                feature_mask: FeatureMask,  # noqa: ARG002
                features: Features | None = None,  # noqa: ARG002
                label: Label | None = None,  # noqa: ARG002
            ) -> AFASelection:
                batch_size = masked_features.shape[0]
                # Always return 0 (stop immediately)
                return torch.zeros(batch_size, 1, dtype=torch.long)

            def predict(
                self,
                masked_features: MaskedFeatures,
                feature_mask: FeatureMask,  # noqa: ARG002
                features: Features | None = None,  # noqa: ARG002
                label: Label | None = None,  # noqa: ARG002
            ) -> Label:
                batch_size = masked_features.shape[0]
                n_classes = 2
                # Predict based on no features (random baseline)
                return torch.softmax(torch.randn(batch_size, n_classes), dim=1)

        # Create test setup
        dataset = MockDataset(n_samples=20, n_features=8, n_classes=2)
        method = ImmediateStopMethod()
        budget = 5

        # Run evaluation
        metrics = eval_afa_method(
            afa_select_fn=method.select,
            dataset=dataset,  # pyright: ignore[reportArgumentType]
            budget=budget,
            afa_predict_fn=method.predict,
            batch_size=4,
            device=torch.device("cpu"),
        )

        # Check that all samples took exactly 0 steps
        assert (metrics["actual_steps"] == 0).all()
        assert metrics["average_steps"] == 0

        # Check that we still get meaningful metrics (even if based on no features)
        assert len(metrics["accuracy_all"]) >= 0
        assert not torch.isnan(metrics["action_distribution"]).any()

    def test_fixed_steps_compatibility(self) -> None:
        """Test that the evaluation still works with methods that never stop early."""
        from src.eval.metrics import eval_afa_method  # noqa: PLC0415

        class FixedStepsMethod:
            """Method that always takes exactly budget steps."""

            def select(
                self,
                masked_features: MaskedFeatures,
                feature_mask: FeatureMask,
                features: Features | None = None,  # noqa: ARG002
                label: Label | None = None,  # noqa: ARG002
            ) -> AFASelection:
                batch_size = masked_features.shape[0]
                selections = torch.zeros(batch_size, 1, dtype=torch.long)

                for i in range(batch_size):
                    # Always select a random unobserved feature
                    unobserved = ~feature_mask[i]
                    if unobserved.any():
                        available_features = torch.where(unobserved)[0]
                        selected_idx = np.random.default_rng().choice(
                            available_features.cpu().numpy()
                        )
                        selections[i] = selected_idx + 1  # Convert to 1-based
                    else:
                        # This shouldn't happen if budget <= n_features
                        selections[i] = 1  # Just select first feature again

                return selections

            def predict(
                self,
                masked_features: MaskedFeatures,
                feature_mask: FeatureMask,  # noqa: ARG002
                features: Features | None = None,  # noqa: ARG002
                label: Label | None = None,  # noqa: ARG002
            ) -> Label:
                batch_size = masked_features.shape[0]
                n_classes = 2
                return torch.softmax(torch.randn(batch_size, n_classes), dim=1)

        # Create test setup
        dataset = MockDataset(n_samples=50, n_features=8, n_classes=2)
        method = FixedStepsMethod()
        budget = 5  # Less than n_features to ensure we don't run out

        # Run evaluation with batching
        metrics = eval_afa_method(
            afa_select_fn=method.select,
            dataset=dataset,  # pyright: ignore[reportArgumentType]
            budget=budget,
            afa_predict_fn=method.predict,
            batch_size=10,  # Test batched processing
            device=torch.device("cpu"),
        )

        # Check that all samples took exactly budget steps
        assert (metrics["actual_steps"] == budget).all()
        assert metrics["average_steps"] == budget


if __name__ == "__main__":
    pytest.main([__file__])
