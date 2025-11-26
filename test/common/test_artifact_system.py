import json
import shutil
import tempfile
from pathlib import Path

import pytest

from afabench.common.registry import get_afa_initializer, get_afa_unmasker
from afabench.common.utils import (
    load_artifact_metadata,
    load_dataset_artifact,
    load_initializer,
    load_unmasker,
)


@pytest.fixture
def temp_dir():
    d = Path(tempfile.mkdtemp())
    yield d
    shutil.rmtree(d)


class TestUnmaskerRegistry:
    def test_one_based_index_returns_callable(self):
        fn = get_afa_unmasker("one_based_index")
        assert callable(fn)

    def test_image_patch_returns_callable(self):
        fn = get_afa_unmasker(
            "image_patch", image_side_length=28, n_channels=1, patch_size=7
        )
        assert callable(fn)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown unmasker"):
            get_afa_unmasker("invalid")

    def test_missing_kwargs_raises(self):
        with pytest.raises(KeyError):
            get_afa_unmasker("image_patch")


class TestInitializerRegistry:
    def test_zero(self):
        init = get_afa_initializer("zero")
        assert init.select_features(20, 5) == []

    def test_fixed_random_cached(self):
        init = get_afa_initializer("fixed_random", seed=42)
        s1 = init.select_features(20, 5)
        s2 = init.select_features(20, 5)
        assert len(s1) == 5
        assert s1 == s2

    def test_fixed_random_seed_matters(self):
        s1 = get_afa_initializer("fixed_random", seed=1).select_features(20, 5)
        s2 = get_afa_initializer("fixed_random", seed=2).select_features(20, 5)
        assert s1 != s2

    def test_random_per_episode_varies(self):
        init = get_afa_initializer("random_per_episode", seed=42)
        results = [tuple(init.select_features(20, 3)) for _ in range(10)]
        assert len(set(results)) > 1

    def test_manual(self):
        init = get_afa_initializer("manual", feature_indices=[0, 5, 10])
        assert init.select_features(20, 3) == [0, 5, 10]

    def test_manual_wrong_count_raises(self):
        init = get_afa_initializer("manual", feature_indices=[0, 5])
        with pytest.raises(AssertionError):
            init.select_features(20, 3)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown initializer"):
            get_afa_initializer("invalid")


class TestLoadUnmaskerInitializer:
    def test_load_unmasker(self):
        fn = load_unmasker("one_based_index")
        assert callable(fn)

    def test_load_initializer(self):
        init = load_initializer("zero")
        assert init.select_features(10, 3) == []


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
