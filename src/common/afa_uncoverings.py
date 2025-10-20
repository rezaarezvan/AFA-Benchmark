"""Different types of AFAUncoverFn."""

from common.custom_types import (
    AFASelection,
    FeatureMask,
    Features,
    MaskedFeatures,
)


def one_based_index_uncovering(
    masked_features: MaskedFeatures,  # noqa: ARG001
    feature_mask: FeatureMask,
    features: Features,
    afa_selection: AFASelection,
) -> tuple[MaskedFeatures, FeatureMask]:
    """
    Uncover the features assuming `afa_selection` are 1-based indices of the features to uncover.

    afa_selection == 0 is ignored.
    """
    relevant_indices = afa_selection != 0
    feature_indices = afa_selection[relevant_indices] - 1

    new_feature_mask = feature_mask.clone()
    new_feature_mask[relevant_indices.nonzero().flatten(), feature_indices] = (
        True
    )
    new_masked_features = features.clone()
    new_masked_features.masked_fill_(~new_feature_mask, 0)

    return new_feature_mask, new_masked_features
