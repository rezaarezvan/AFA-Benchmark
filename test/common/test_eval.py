import torch

from afabench.common.custom_types import (
    AFASelection,
    FeatureMask,
    Features,
    MaskedFeatures,
)
from afabench.eval.eval import single_afa_step


def afa_unmask_fn(
    masked_features: MaskedFeatures,
    feature_mask: FeatureMask,
    features: Features,
    afa_selection: AFASelection,
) -> tuple[FeatureMask, MaskedFeatures]:
    # 6 features but selection is 1-3. Unmask a "block" of features.
    batch_size, num_features = masked_features.shape
    new_feature_mask = feature_mask.clone()
    for i in range(batch_size):
        selection = int(afa_selection[i].item())
        if selection > 0:
            start_idx = (selection - 1) * 2
            end_idx = min(start_idx + 2, num_features)
            new_feature_mask[i, start_idx:end_idx] = 1

    new_masked_features = features * new_feature_mask
    return new_feature_mask, new_masked_features


def test_single_afa_step() -> None:
    """Test that single_afa_step works when the selections are not 0."""
    features = torch.tensor(
        [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]], dtype=torch.float32
    )
    masked_features = torch.tensor(
        [[1, 2, 0, 0, 0, 0], [0, 0, 9, 10, 0, 0]], dtype=torch.float32
    )
    feature_mask = torch.tensor(
        [[1, 1, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0]], dtype=torch.bool
    )

    def afa_select_fn(
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,  # noqa: ARG001
    ) -> AFASelection:
        # Always output selection 3
        return 3 * torch.ones((masked_features.shape[0], 1), dtype=torch.int64)

    selection, new_masked_features, new_feature_mask, _, _ = single_afa_step(
        features=features,
        masked_features=masked_features,
        feature_mask=feature_mask,
        afa_select_fn=afa_select_fn,
        afa_unmask_fn=afa_unmask_fn,
    )

    assert torch.allclose(
        selection, torch.tensor([[3], [3]], dtype=torch.int64)
    ), f"Expected selection [[3], [3]], but got {selection}"
    assert torch.allclose(
        new_masked_features,
        torch.tensor(
            [[1, 2, 0, 0, 5, 6], [0, 0, 9, 10, 11, 12]], dtype=torch.float32
        ),
    ), (
        f"Expected new masked features [[1, 2, 0, 0, 5, 6], [0, 0, 9, 10, 11, 12]], but got {new_masked_features}"
    )
    assert torch.allclose(
        new_feature_mask,
        torch.tensor(
            [[1, 1, 0, 0, 1, 1], [0, 0, 1, 1, 1, 1]], dtype=torch.bool
        ),
    ), (
        f"Expected new feature mask [[1, 1, 0, 0, 1, 1], [0, 0, 1, 1, 1, 1]], but got {new_feature_mask}"
    )


def test_single_afa_step_stop_selection() -> None:
    """Test that single_afa_step works when some selections are 0."""
    features = torch.tensor(
        [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]], dtype=torch.float32
    )
    masked_features = torch.tensor(
        [[1, 2, 0, 0, 0, 0], [0, 0, 9, 10, 0, 0]], dtype=torch.float32
    )
    feature_mask = torch.tensor(
        [[1, 1, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0]], dtype=torch.bool
    )

    def afa_select_fn(
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,  # noqa: ARG001
    ) -> AFASelection:
        # Output selection 0 if the first feature is observed, otherwise 3
        batch_size = masked_features.shape[0]
        selections = 3 * torch.ones((batch_size, 1), dtype=torch.int64)
        for i in range(batch_size):
            if masked_features[i, 0] == 1:
                selections[i] = 0
        return selections

    selection, new_masked_features, new_feature_mask, _, _ = single_afa_step(
        features=features,
        masked_features=masked_features,
        feature_mask=feature_mask,
        afa_select_fn=afa_select_fn,
        afa_unmask_fn=afa_unmask_fn,
    )

    assert torch.allclose(
        selection, torch.tensor([[0], [3]], dtype=torch.int64)
    ), f"Expected selection [[0], [3]], but got {selection}"
    assert torch.allclose(
        new_masked_features,
        torch.tensor(
            [[1, 2, 0, 0, 0, 0], [0, 0, 9, 10, 11, 12]], dtype=torch.float32
        ),
    ), (
        f"Expected new masked features [[1, 2, 0, 0, 0, 0], [0, 0, 9, 10, 11, 12]], but got {new_masked_features}"
    )
    assert torch.allclose(
        new_feature_mask,
        torch.tensor(
            [[1, 1, 0, 0, 0, 0], [0, 0, 1, 1, 1, 1]], dtype=torch.bool
        ),
    ), (
        f"Expected new feature mask [[1, 1, 0, 0, 0, 0], [0, 0, 1, 1, 1, 1]], but got {new_feature_mask}"
    )
