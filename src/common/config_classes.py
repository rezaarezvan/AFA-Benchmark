from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore

# @dataclass
# class DatasetConfig:
#     class_name: str # e.g., "AFAContext"


@dataclass
class Shim2018EncoderConfig:
    output_size: int = 16
    reading_block_cells: tuple[int, ...] = (32, 32)
    writing_block_cells: tuple[int, ...] = (32, 32)
    memory_size: int = 16
    processing_steps: int = 5
    dropout: float = 0.1


@dataclass
class Shim2018ClassifierConfig:
    num_cells: tuple[int, ...] = (32, 32)


@dataclass
class DatasetConfig:
    artifact_name: str  # e.g "afacontext:may26"


@dataclass
class Shim2018PretrainConfig:
    dataset: DatasetConfig
    batch_size: int  # batch size for dataloader
    epochs: int

    device: str = "cuda"
    seed: int = 42
    lr: float = 1e-3
    max_masking_probability: float = 0.9
    encoder: Shim2018EncoderConfig = field(default_factory=Shim2018EncoderConfig)
    classifier: Shim2018ClassifierConfig = field(
        default_factory=Shim2018ClassifierConfig
    )


cs = ConfigStore.instance()
cs.store(name="shim2018_pretrain", node=Shim2018PretrainConfig)
