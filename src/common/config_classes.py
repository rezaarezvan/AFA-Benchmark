from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Literal
from hydra.core.config_store import ConfigStore

cs = ConfigStore.instance()


@dataclass
class SplitRatioConfig:
    train: float = 0.7  # ratio of training data
    val: float = 0.15  # ratio of validation data
    test: float = 0.15  # ratio of test data


class DatasetType(str, Enum):
    cube = "cube"
    MNIST = "MNIST"
    shim2018cube = "shim2018cube"
    diabetes = "diabetes"
    physionet = "physionet"
    miniboone = "miniboone"
    FashionMNIST = "FashionMNIST"


@dataclass
class DatasetConfig:
    type: DatasetType
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetGenerationConfig:
    data_dir: str = (
        "data"  # where to store the generated dataset (apart from wandb artifacts)
    )
    split_idx: int = 1  # which split to use
    seeds: list[int] = field(
        default_factory=lambda: [
            42,
            123,
            456,
            789,
            101112,
            131415,
            161718,
            192021,
            222324,
            252627,
        ]
    )  # which seeds to use, only the seed at index `split - 1` will be used
    split_ratio: SplitRatioConfig = field(default_factory=SplitRatioConfig)
    output_artifact_aliases: list[str] = field(default_factory=lambda: [])

    dataset: DatasetConfig = field(default_factory=lambda: DatasetConfig("cube"))


cs.store(name="dataset_generation", node=DatasetGenerationConfig)


@dataclass
class Shim2018EncoderConfig:
    output_size: int = 16
    reading_block_cells: list[int] = field(default_factory=lambda: [32, 32])
    writing_block_cells: list[int] = field(default_factory=lambda: [32, 32])
    memory_size: int = 16
    processing_steps: int = 5
    dropout: float = 0.1


@dataclass
class Shim2018ClassifierConfig:
    num_cells: list[int] = field(default_factory=lambda: [32, 32])


@dataclass
class Shim2018PretrainConfig:
    dataset_artifact_name: str
    batch_size: int  # batch size for dataloader
    epochs: int
    limit_train_batches: int | None = None
    limit_val_batches: int | None = None

    device: str = "cuda"
    seed: int = 42
    lr: float = 1e-3
    min_masking_probability: float = 0.0
    max_masking_probability: float = 0.9
    encoder: Shim2018EncoderConfig = field(default_factory=Shim2018EncoderConfig)
    classifier: Shim2018ClassifierConfig = field(
        default_factory=Shim2018ClassifierConfig
    )
    output_artifact_aliases: list[str] = field(default_factory=lambda: [])


cs.store(name="pretrain_shim2018", node=Shim2018PretrainConfig)


@dataclass
class Shim2018AgentConfig:
    eps_steps: int
    replay_buffer_size: int
    replay_buffer_batch_size: int
    num_optim: int
    init_random_frames: int = 0  # how many frames to collect before starting training
    eps_init: float = 1.0
    eps_end: float = 0.1
    lr: float = 1e-3
    update_tau: float = 0.005
    max_grad_norm: float = 1.0
    replay_buffer_alpha: float = 0.6
    replay_buffer_beta_init: float = 0.4
    replay_buffer_beta_end: float = 1.0
    delay_value: bool = True
    double_dqn: bool = True
    action_value_num_cells: list[int] = field(default_factory=lambda: [128, 128])


@dataclass
class Shim2018TrainConfig:
    pretrained_model_artifact_name: str
    n_agents: int
    hard_budget: int
    agent: Shim2018AgentConfig
    n_batches: int  # how many batches to train the agent
    batch_size: int  # batch size for collector
    eval_every_n_batches: int  # how often to evaluate the agent
    eval_max_steps: int  # maximum allowed number of steps in an evaluation episode
    n_eval_episodes: int  # how many episodes to average over in evaluation

    device: str = "cuda"
    seed: int = 42
    pretrained_model_lr: float = 1e-3
    activate_joint_training_after_n_batches: int = 0
    output_artifact_aliases: list[str] = field(default_factory=lambda: [])
    evaluate_final_performance: bool = True
    eval_only_n_samples: int | None = None


cs.store(name="train_shim2018", node=Shim2018TrainConfig)


@dataclass
class Ma2018PointNetConfig:
    identity_size: int = 20
    identity_network_num_cells: list[int] = field(default_factory=lambda: [20, 20])
    output_size: int = 40
    feature_map_encoder_num_cells: list[int] = field(default_factory=lambda: [500])


@dataclass
class Ma2018PartialVAEConfig:
    lr: float = 1e-3
    epochs: int = 1000
    patience: int = 5
    encoder_num_cells: list[int] = field(default_factory=lambda: [500, 500, 200])
    latent_size: int = 20
    kl_scaling_factor: float = 0.1
    max_masking_probability: float = 0.9
    decoder_num_cells: list[int] = field(default_factory=lambda: [200, 500, 500])


@dataclass
class Ma2018ClassifierConfig:
    lr: float = 1e-3
    epochs: int = 100
    patience: int = 5


@dataclass
class Ma2018PretrainConfig:
    dataset_artifact_name: str
    pretrained_model_path: str = "./models/ma2018"
    output_artifact_aliases: list[str] = field(default_factory=lambda: [])

    batch_size: int = 128
    seed: int = 42
    device: str = "cuda"

    pointnet: Ma2018PointNetConfig = field(default_factory=Ma2018PointNetConfig)
    partial_vae: Ma2018PartialVAEConfig = field(default_factory=Ma2018PartialVAEConfig)
    classifier: Ma2018ClassifierConfig = field(default_factory=Ma2018ClassifierConfig)


cs.store(name="pretrain_ma2018", node=Ma2018PretrainConfig)


@dataclass
class RandomDummyTrainConfig:
    dataset_artifact_name: str
    hard_budget: int  # not used, but pretend that it is
    seed: int = 42
    output_artifact_aliases: list[str] = field(default_factory=lambda: [])


cs.store(name="train_randomdummy", node=RandomDummyTrainConfig)


@dataclass
class TrainMaskedMLPClassifierConfig:
    dataset_artifact_name: str
    batch_size: int
    epochs: int
    limit_train_batches: int | None = None
    limit_val_batches: int | None = None
    min_masking_probability: float = 0.0
    max_masking_probability: float = 0.9
    eval_only_n_samples: int | None = (
        None  # if specified, only evaluate on this many samples
    )
    lr: float = 1e-3
    seed: int = 42
    device: str = "cuda"
    num_cells: list[int] = field(default_factory=lambda: [128, 128])
    dropout: float = 0.1
    output_artifact_aliases: list[str] = field(default_factory=lambda: [])
    evaluate_final_performance: bool = True


cs.store(name="train_masked_mlp_classifier", node=TrainMaskedMLPClassifierConfig)


@dataclass
class EvalConfig:
    trained_method_artifact_name: str
    trained_classifier_artifact_name: str | None  # if None, use the method's classifier
    seed: int = 42
    device: str = "cuda"
    output_artifact_aliases: list[str] = field(default_factory=lambda: [])
    eval_only_n_samples: int | None = (
        None  # if specified, only evaluate on this many samples
    )
    batch_size: int = 1


cs.store(name="eval", node=EvalConfig)


@dataclass
class MetricConfig:
    key: str
    description: str
    ylim: list[int] | None


@dataclass
class PlotConfig:
    eval_artifact_names: list[str]
    metric_keys_and_descriptions: list[
        MetricConfig
    ]  # Inner list has two elements: [metric_key, description]


cs.store(name="plot", node=PlotConfig)
