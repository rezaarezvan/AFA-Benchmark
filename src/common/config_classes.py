from dataclasses import dataclass, field

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
class Shim2018PretrainConfig:
    seed: int
    dataset_artifact_name: str  # wandb artifact name, e.g., "afa_context:may13"
    batch_size: int  # batch size for dataloader
    encoder: Shim2018EncoderConfig = field(default_factory=Shim2018EncoderConfig)
    classifier: Shim2018ClassifierConfig = field(
        default_factory=Shim2018ClassifierConfig
    )
    lr: float = 1e-3
    max_masking_probability: float = 0.9
    device: str = "cuda"
    epochs: int = 100
