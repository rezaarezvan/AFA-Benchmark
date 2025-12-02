from afabench.common.config_classes import (
    AACODefaultInitializerConfig,
    FixedRandomInitializerConfig,
    InitializerConfig,
    LeastInformativeInitializerConfig,
    ManualInitializerConfig,
    MutualInformationInitializerConfig,
    RandomPerEpisodeInitializerConfig,
)
from afabench.common.custom_types import AFAInitializer
from afabench.common.initializers.aaco_default_initializer import (
    AACODefaultInitializer,
)
from afabench.common.initializers.dynamic_random_initializer import (
    DynamicRandomInitializer,
)
from afabench.common.initializers.fixed_random_initializer import (
    FixedRandomInitializer,
)
from afabench.common.initializers.least_informative_initializer import (
    LeastInformativeInitializer,
)
from afabench.common.initializers.manual_initializer import ManualInitializer
from afabench.common.initializers.mutual_information_initializer import (
    MutualInformationInitializer,
)
from afabench.common.initializers.zero_initializer import ZeroInitializer
from afabench.common.registry import get_afa_initializer_class


def get_afa_initializer_from_config(
    initializer_config: InitializerConfig,
) -> AFAInitializer:
    """Get initializer from config."""
    if initializer_config.class_name == "ZeroInitializer":
        assert initializer_config.config is None

        cls = get_afa_initializer_class(initializer_config.class_name)
        assert cls is ZeroInitializer
        return cls()

    if initializer_config.class_name == "FixedRandomInitializer":
        assert isinstance(
            initializer_config.config, FixedRandomInitializerConfig
        )

        cls = get_afa_initializer_class(initializer_config.class_name)
        assert cls is FixedRandomInitializer
        return cls(config=initializer_config.config)

    if initializer_config.class_name == "DynamicRandomInitializer":
        assert isinstance(
            initializer_config.config, RandomPerEpisodeInitializerConfig
        )

        cls = get_afa_initializer_class(initializer_config.class_name)
        assert cls is DynamicRandomInitializer
        return cls(config=initializer_config.config)

    if initializer_config.class_name == "ManualInitializer":
        assert isinstance(initializer_config.config, ManualInitializerConfig)

        cls = get_afa_initializer_class(initializer_config.class_name)
        assert cls is ManualInitializer
        return cls(config=initializer_config.config)

    if initializer_config.class_name == "MutualInformationInitializer":
        assert isinstance(
            initializer_config.config, MutualInformationInitializerConfig
        )

        cls = get_afa_initializer_class(initializer_config.class_name)
        assert cls is MutualInformationInitializer
        return cls(config=initializer_config.config)

    if initializer_config.class_name == "LeastInformativeInitializer":
        assert isinstance(
            initializer_config.config, LeastInformativeInitializerConfig
        )

        cls = get_afa_initializer_class(initializer_config.class_name)
        assert cls is LeastInformativeInitializer
        return cls(config=initializer_config.config)

    if initializer_config.class_name == "AACODefaultInitializer":
        assert isinstance(
            initializer_config.config, AACODefaultInitializerConfig
        )

        cls = get_afa_initializer_class(initializer_config.class_name)
        assert cls is AACODefaultInitializer
        return cls(config=initializer_config.config)

    msg = f"Unknown initializer: {initializer_config.class_name}"
    raise ValueError(msg)
