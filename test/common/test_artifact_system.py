import json
import shutil
import tempfile
from pathlib import Path

import pytest
import torch

from afabench.common.config_classes import (
    FixedRandomInitializerConfig,
    InitializerConfig,
    ManualInitializerConfig,
    RandomPerEpisodeInitializerConfig,
    UnmaskerConfig,
)
from afabench.common.initializers.utils import get_afa_initializer_from_config
from afabench.common.unmaskers.utils import get_afa_unmasker_from_config
from afabench.common.utils import (
    load_artifact_metadata,
    load_dataset_artifact,
)


@pytest.fixture
def temp_dir():
    d = Path(tempfile.mkdtemp())
    yield d
    shutil.rmtree(d)


class TestUnmaskerRegistry:
    def test_direct_unmasker_returns_unmasker(self):
        config = UnmaskerConfig(class_name="DirectUnmasker", config=None)
        unmasker = get_afa_unmasker_from_config(config)
        assert hasattr(unmasker, "unmask")
        assert hasattr(unmasker, "get_n_selections")

    def test_image_patch_unmasker_returns_unmasker(self):
        from afabench.common.config_classes import ImagePatchUnmaskerConfig

        patch_config = ImagePatchUnmaskerConfig(
            image_side_length=28, n_channels=1, patch_size=7
        )
        config = UnmaskerConfig(
            class_name="ImagePatchUnmasker", config=patch_config
        )
        unmasker = get_afa_unmasker_from_config(config)
        assert hasattr(unmasker, "unmask")
        assert hasattr(unmasker, "get_n_selections")

    def test_unknown_unmasker_raises(self):
        config = UnmaskerConfig(class_name="InvalidUnmasker", config=None)
        with pytest.raises(ValueError, match="Unknown unmasker"):
            get_afa_unmasker_from_config(config)


class TestInitializerRegistry:
    def test_zero_initializer(self):
        config = InitializerConfig(class_name="ZeroInitializer", config=None)
        initializer = get_afa_initializer_from_config(config)

        # Test with some dummy data
        features = torch.randn(5, 3, 4)
        feature_shape = torch.Size([3, 4])
        mask = initializer.initialize(
            features=features, feature_shape=feature_shape
        )

        # ZeroInitializer should return all False mask
        assert mask.shape == features.shape
        assert not mask.any()  # All should be False

    def test_fixed_random_initializer(self):
        config_data = FixedRandomInitializerConfig(unmask_ratio=0.3)
        config = InitializerConfig(
            class_name="FixedRandomInitializer", config=config_data
        )
        initializer = get_afa_initializer_from_config(config)

        # Test with some dummy data
        features = torch.randn(5, 3, 4)
        feature_shape = torch.Size([3, 4])

        initializer.set_seed(42)
        mask1 = initializer.initialize(
            features=features, feature_shape=feature_shape
        )
        mask2 = initializer.initialize(
            features=features, feature_shape=feature_shape
        )

        # Fixed random should be cached and return same results
        assert torch.equal(mask1, mask2)

        # Should respect unmask_ratio approximately
        expected_count = int(feature_shape.numel() * 0.3)
        actual_count = mask1[0].sum().item()  # Count for first batch element
        assert actual_count == expected_count

    def test_dynamic_random_initializer(self):
        config_data = RandomPerEpisodeInitializerConfig(unmask_ratio=0.25)
        config = InitializerConfig(
            class_name="DynamicRandomInitializer", config=config_data
        )
        initializer = get_afa_initializer_from_config(config)

        # Test with some dummy data
        features = torch.randn(10, 4, 3)
        feature_shape = torch.Size([4, 3])

        initializer.set_seed(42)
        mask = initializer.initialize(
            features=features, feature_shape=feature_shape
        )

        # Should respect unmask_ratio
        expected_count = int(feature_shape.numel() * 0.25)
        actual_count = mask[0].sum().item()  # Count for first batch element
        assert actual_count == expected_count

    def test_manual_initializer(self):
        config_data = ManualInitializerConfig(flat_feature_indices=[0, 5, 10])
        config = InitializerConfig(
            class_name="ManualInitializer", config=config_data
        )
        initializer = get_afa_initializer_from_config(config)

        # Test with some dummy data that has at least 11 features
        features = torch.randn(5, 4, 4)  # 16 features total
        feature_shape = torch.Size([4, 4])
        mask = initializer.initialize(
            features=features, feature_shape=feature_shape
        )

        # Should have exactly 3 features selected
        assert mask[0].sum().item() == 3

        # Check that the right indices are selected
        flat_mask = mask[0].flatten()
        assert flat_mask[0].item() == True
        assert flat_mask[5].item() == True
        assert flat_mask[10].item() == True

    def test_unknown_initializer_raises(self):
        config = InitializerConfig(
            class_name="InvalidInitializer", config=None
        )
        with pytest.raises(ValueError, match="Unknown initializer"):
            get_afa_initializer_from_config(config)


class TestArtifactMetadata:
    def test_load_metadata(self, temp_dir):
        path = temp_dir / "artifact"
        path.mkdir()
        (path / "metadata.json").write_text(json.dumps({"key": "value"}))
        meta = load_artifact_metadata(path)
        assert meta == {"key": "value"}

    def test_missing_metadata_raises(self, temp_dir):
        empty = temp_dir / "empty"
        empty.mkdir()
        with pytest.raises(FileNotFoundError):
            load_artifact_metadata(empty)

    def test_corrupted_json_raises(self, temp_dir):
        bad = temp_dir / "bad"
        bad.mkdir()
        (bad / "metadata.json").write_text("not json")
        with pytest.raises(json.JSONDecodeError):
            load_artifact_metadata(bad)

    def test_invalid_split_raises(self, temp_dir):
        path = temp_dir / "dataset"
        path.mkdir()
        (path / "metadata.json").write_text(
            json.dumps({"dataset_type": "cube"})
        )
        with pytest.raises(ValueError, match="Invalid split"):
            load_dataset_artifact(path, "invalid")


# TODO: Fix and enable these integration tests with real artifacts.
# @pytest.mark.integration
# class TestIntegrationWithRealArtifacts:
#     """Integration tests using real artifacts from extra/ directory."""
#
#     DATASET_PATH = Path("extra/data/cube/cube_split_1")
#     # adjust as needed
#     METHOD_PATH = Path("extra/result/randomdummy/train/cube_split_1")
#     CLASSIFIER_PATH = Path(
#         "extra/classifiers/masked_mlp/cube_split_1")  # adjust as needed
#
#     @pytest.fixture
#     def skip_if_no_dataset(self):
#         if not self.DATASET_PATH.exists():
#             pytest.skip(f"Dataset not found: {self.DATASET_PATH}")
#
#     @pytest.fixture
#     def skip_if_no_method(self):
#         if not self.METHOD_PATH.exists():
#             pytest.skip(f"Method not found: {self.METHOD_PATH}")
#
#     @pytest.fixture
#     def skip_if_no_classifier(self):
#         if not self.CLASSIFIER_PATH.exists():
#             pytest.skip(f"Classifier not found: {self.CLASSIFIER_PATH}")
#
#     def test_load_dataset_artifact(self, skip_if_no_dataset):
#         ds = load_dataset_artifact(self.DATASET_PATH, "train")
#         assert hasattr(ds, "features")
#         assert hasattr(ds, "labels")
#         assert len(ds) > 0
#
#     def test_load_dataset_splits(self, skip_if_no_dataset):
#         train, val, test, meta = load_dataset_splits(self.DATASET_PATH)
#         assert len(train) > 0
#         assert len(val) > 0
#         assert len(test) > 0
#         assert "dataset_type" in meta
#
#     def test_load_method_artifact(self, skip_if_no_method):
#         method = load_method_artifact(self.METHOD_PATH)
#         assert hasattr(method, "select")
#
#     def test_load_classifier_artifact(self, skip_if_no_classifier):
#         clf = load_classifier_artifact(self.CLASSIFIER_PATH)
#         assert callable(clf)
#
#     def test_load_eval_components(self, skip_if_no_dataset, skip_if_no_method, skip_if_no_classifier):
#         method, unmasker, initializer, dataset, classifier = load_eval_components(
#             method_artifact_path=self.METHOD_PATH,
#             unmasker_name="one_based_index",
#             initializer_name="zero",
#             dataset_artifact_path=self.DATASET_PATH,
#             dataset_split="test",
#             classifier_artifact_path=self.CLASSIFIER_PATH,
#         )
#         assert method is not None
#         assert callable(unmasker)
#         assert initializer is not None
#         assert len(dataset) > 0
#         assert classifier is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "not integration"])
