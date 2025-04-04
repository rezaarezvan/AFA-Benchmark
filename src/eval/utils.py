import torch

from common.custom_types import AFADataset, AFAMethod, FeatureMask


def verify_afa_method(afa_method: AFAMethod, dataset: AFADataset):
    """Quickly check that the afa_method does not crash with a specific dataset."""

    features, _ = dataset[0]

    # We always expect a batch dimension
    features = features.unsqueeze(0)

    feature_mask: FeatureMask = torch.ones_like(
        features, dtype=torch.bool, device=features.device
    )

    # Check that the method does not crash
    afa_method(features, feature_mask)
