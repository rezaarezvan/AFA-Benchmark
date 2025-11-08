import torch

from common.afa_uncoverings import (
    get_image_patch_uncover_fn,
    one_based_index_uncover_fn,
)


def test_one_based_index_uncovering() -> None:
    features = torch.tensor(
        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
        dtype=torch.float,
    )
    feature_mask = torch.tensor(
        [[0, 1, 1, 0], [1, 0, 0, 1], [1, 0, 0, 0], [0, 0, 0, 0]],
        dtype=torch.bool,
    )
    masked_features = features.masked_fill(~feature_mask, 0)

    afa_selection = torch.tensor([1, 0, 4, 2])

    new_masked_features, new_feature_mask = one_based_index_uncover_fn(
        masked_features=masked_features,
        feature_mask=feature_mask,
        features=features,
        afa_selection=afa_selection,
    )

    expected_new_feature_mask = torch.tensor(
        [[1, 1, 1, 0], [1, 0, 0, 1], [1, 0, 0, 1], [0, 1, 0, 0]],
        dtype=torch.bool,
    )

    expected_new_masked_features = features.masked_fill(
        ~expected_new_feature_mask, 0
    )

    assert torch.allclose(
        new_feature_mask,
        expected_new_feature_mask,
    ), (
        f"Expected new feature mask to be {expected_new_feature_mask}, but got {new_feature_mask}"
    )

    assert torch.allclose(new_masked_features, expected_new_masked_features), (
        f"Expected new masked features to be {expected_new_masked_features}, but got {new_masked_features}"
    )


def test_image_patch_uncovering() -> None:
    # batch size 1, 2 channels, 4x4 image with patch size 2

    # 0011
    # 0011
    # 1100
    # 1100
    # Converted into (using selection=4, fourth patch)
    # 0011
    # 0011
    # 1111
    # 1111

    feature_mask = (
        torch.tensor(
            [[0, 0, 1, 1], [0, 0, 1, 1], [1, 1, 0, 0], [1, 1, 0, 0]],
            dtype=torch.bool,
        )
        .unsqueeze(0)
        .unsqueeze(0)
        .expand(-1, 2, -1, -1)
    ).flatten(start_dim=1)
    features = torch.arange(2 * 4 * 4, dtype=torch.float).reshape(1, -1)
    masked_features = features.masked_fill(~feature_mask, 0)

    afa_selection = torch.tensor([4])

    image_patch_uncover_fn = get_image_patch_uncover_fn(
        image_side_length=4, n_channels=2, patch_size=2
    )

    new_masked_features, new_feature_mask = image_patch_uncover_fn(
        masked_features=masked_features,
        feature_mask=feature_mask,
        features=features,
        afa_selection=afa_selection,
    )

    expected_new_feature_mask = (
        torch.tensor(
            [[0, 0, 1, 1], [0, 0, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
            dtype=torch.bool,
        )
        .unsqueeze(0)
        .unsqueeze(0)
        .expand(-1, 2, -1, -1)
    ).flatten(start_dim=1)

    expected_new_masked_features = features.masked_fill(
        ~expected_new_feature_mask, 0
    )

    assert torch.allclose(
        new_feature_mask,
        expected_new_feature_mask,
    ), (
        f"Expected new feature mask to be {expected_new_feature_mask}, but got {new_feature_mask}"
    )

    assert torch.allclose(new_masked_features, expected_new_masked_features), (
        f"Expected new masked features to be {expected_new_masked_features}, but got {new_masked_features}"
    )
