import torch

from common.afa_uncoverings import one_based_index_uncovering


def test_one_based_index_uncovering() -> None:
    feature_mask = torch.tensor(
        [[0, 1, 1, 0], [1, 0, 0, 1], [1, 0, 0, 0], [0, 0, 0, 0]],
        dtype=torch.bool,
    )
    features = torch.tensor(
        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
        dtype=torch.float,
    )
    masked_features = torch.tensor(
        [[0, 2, 3, 0], [5, 0, 0, 8], [9, 0, 0, 0], [0, 0, 0, 0]],
        dtype=torch.float,
    )

    afa_selection = torch.tensor([1, 0, 4, 2])

    new_feature_mask, new_masked_features = one_based_index_uncovering(
        masked_features=masked_features,
        feature_mask=feature_mask,
        features=features,
        afa_selection=afa_selection,
    )

    expected_new_feature_mask = torch.tensor(
        [[1, 1, 1, 0], [1, 0, 0, 1], [1, 0, 0, 1], [0, 1, 0, 0]],
        dtype=torch.bool,
    )

    expected_new_masked_features = torch.tensor(
        [[1, 2, 3, 0], [5, 0, 0, 8], [9, 0, 0, 12], [0, 14, 0, 0]],
        dtype=torch.float,
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
