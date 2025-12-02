from afabench.common.config_classes import (
    ImagePatchUnmaskerConfig,
    UnmaskerConfig,
)
from afabench.common.custom_types import AFAUnmasker
from afabench.common.registry import get_afa_unmasker_class
from afabench.common.unmaskers.direct_unmasker import DirectUnmasker
from afabench.common.unmaskers.image_patch_unmasker import ImagePatchUnmasker


def get_afa_unmasker_from_config(
    unmasker_config: UnmaskerConfig,
) -> AFAUnmasker:
    """Get unmasker function by config."""
    if unmasker_config.class_name == "DirectUnmasker":
        assert unmasker_config.config is None

        cls = get_afa_unmasker_class(unmasker_config.class_name)
        assert cls is DirectUnmasker
        return cls()

    if unmasker_config.class_name == "ImagePatchUnmasker":
        assert isinstance(unmasker_config.config, ImagePatchUnmaskerConfig)

        cls = get_afa_unmasker_class(unmasker_config.class_name)
        assert cls is ImagePatchUnmasker
        return cls(config=unmasker_config.config)

    msg = f"Unknown unmasker: {unmasker_config.class_name}"
    raise ValueError(msg)
