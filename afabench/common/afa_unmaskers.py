from typing import final, override

import torch
from torch.nn import functional as F

from afabench.common.config_classes import ImagePatchUnmaskerConfig
from afabench.common.custom_types import (
    AFASelection,
    AFAUnmasker,
    FeatureMask,
    Features,
    Label,
    MaskedFeatures,
    SelectionMask,
)


# Implements AFAUnmaskFn
class DirectUnmasker(AFAUnmasker):
    @override
    def set_seed(self, seed: int | None) -> None:
        # This unmasker is deterministic
        pass

    @override
    def unmask(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
        features: Features,
        afa_selection: AFASelection,
        selection_mask: SelectionMask,
        label: Label | None = None,
        feature_shape: torch.Size | None = None,
    ) -> FeatureMask:
        """
        Unmasks the features assuming `afa_selection` are 1-based indices of the features to uncover.

        afa_selection == 0 is ignored.
        """
        assert feature_shape is not None, "feature_shape must be provided"
        assert afa_selection.shape[1] == 1, (
            "AFASelection must have shape [batch, 1]"
        )
        feature_indices = afa_selection.squeeze(1) - 1

        new_feature_mask = feature_mask.clone()
        # Convert feature_indices to n-dimensional indices using unravel_index
        multi_indices = torch.unravel_index(feature_indices, feature_shape)
        print(f"multi_indices = {multi_indices}")
        batch_indices = torch.arange(
            feature_mask.size(0), device=feature_mask.device
        )
        new_feature_mask[batch_indices, *multi_indices] = True

        return new_feature_mask


@final
class ImagePatchUnmasker(AFAUnmasker):
    def __init__(self, config: ImagePatchUnmaskerConfig):
        assert config.image_side_length % config.patch_size == 0, (
            "Image side length must be divisible by patch size"
        )
        self.config = config
        self.low_dim_image_side_length = (
            config.image_side_length // config.patch_size
        )
        self.afa_selection_size = self.low_dim_image_side_length**2

    @override
    def set_seed(self, seed: int | None) -> None:
        # This unmasker is deterministic
        pass

    @override
    def unmask(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
        features: Features,
        afa_selection: AFASelection,
        selection_mask: SelectionMask,
        label: Label | None = None,
        feature_shape: torch.Size | None = None,
    ) -> FeatureMask:
        assert afa_selection.shape[1] == 1, (
            "AFASelection must have shape [batch, 1]"
        )

        # Convert afa selection into 1D mask
        sel = afa_selection.to(torch.long)
        stop = sel == 0
        # avoid -1 for one hot
        sel = torch.where(stop, torch.ones_like(sel), sel)
        afa_selection_1d = F.one_hot(
            sel - 1, num_classes=self.afa_selection_size
        )
        afa_selection_1d[stop] = 0

        # Convert to low-dimensional image mask
        afa_selection_low_dim_image = afa_selection_1d.view(
            -1,
            self.low_dim_image_side_length,
            self.low_dim_image_side_length,
        )

        # Add batch dimension and channel dimension, expand channels
        afa_selection_low_dim_image = afa_selection_low_dim_image.unsqueeze(
            1
        ).expand(-1, self.config.n_channels, -1, -1)

        # Upscale image mask, converting between float and bool
        afa_selection_image = F.interpolate(
            afa_selection_low_dim_image.float(),
            scale_factor=self.config.patch_size,
            mode="nearest-exact",
        ).bool()

        # print(f"{afa_selection_image.shape}")
        # print(f"{feature_mask.shape}")

        # Convert image mask to feature mask and add to previous feature mask
        # new_feature_mask = feature_mask + afa_selection_image.flatten(
        #     start_dim=1
        # )
        new_feature_mask = feature_mask | afa_selection_image

        return new_feature_mask
