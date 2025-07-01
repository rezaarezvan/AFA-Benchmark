from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Literal
from hydra.core.config_store import ConfigStore
from torch import nn

from afa_rl.zannone2019.models import PartialVAELossType, PointNetType

cs = ConfigStore.instance()

# --- DATASETS ---


@dataclass
class SplitRatioConfig:
    train: float  # ratio of training data
    val: float  # ratio of validation data
    test: float  # ratio of test data


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
    data_dir: str  # where to store the generated dataset (apart from wandb artifacts)
    split_idx: int  # which split to use
    seeds: list[
        int
    ]  # which seeds to use, only the seed at index `split - 1` will be used
    split_ratio: SplitRatioConfig
    output_artifact_aliases: list[str]

    dataset: DatasetConfig


cs.store(name="dataset_generation", node=DatasetGenerationConfig)

# --- PRETRAINING MODELS ---

# shim2018


@dataclass
class Shim2018EncoderConfig:
    output_size: int
    reading_block_cells: list[int]
    writing_block_cells: list[int]
    memory_size: int
    processing_steps: int
    dropout: float


@dataclass
class Shim2018ClassifierConfig:
    num_cells: list[int]


@dataclass
class Shim2018PretrainConfig:
    dataset_artifact_name: str
    batch_size: int  # batch size for dataloader
    epochs: int
    limit_train_batches: int | None
    limit_val_batches: int | None

    device: str
    seed: int
    lr: float
    min_masking_probability: float
    max_masking_probability: float
    encoder: Shim2018EncoderConfig
    classifier: Shim2018ClassifierConfig
    output_artifact_aliases: list[str]


cs.store(name="pretrain_shim2018", node=Shim2018PretrainConfig)

# zannone2019


@dataclass
class Zannone2019PointNetConfig:
    type: str  # "pointnet" or "pointnetplus"
    identity_size: int
    max_embedding_norm: float
    output_size: int
    feature_map_encoder_num_cells: list[int]
    feature_map_encoder_dropout: float


@dataclass
class Zannone2019EncoderConfig:
    num_cells: list[int]
    dropout: float


@dataclass
class Zannone2019PartialVAEConfig:
    latent_size: int
    decoder_num_cells: list[int]
    decoder_dropout: float


@dataclass
class Zannone2019ClassifierConfig:
    num_cells: list[int]
    dropout: float


@dataclass
class Zannone2019PretrainConfig:
    dataset_artifact_name: str
    batch_size: int  # batch size for dataloader
    epochs: int
    limit_train_batches: int | None
    limit_val_batches: int | None

    device: str
    seed: int
    lr: float
    min_masking_probability: float
    max_masking_probability: float
    pointnet: Zannone2019PointNetConfig
    encoder: Zannone2019EncoderConfig
    partial_vae: Zannone2019PartialVAEConfig
    classifier: Zannone2019ClassifierConfig
    recon_loss_type: str  # one of "squared_error" or "binary_cross_entropy"
    kl_scaling_factor: float
    classifier_loss_scaling_factor: float
    output_artifact_aliases: list[str]


cs.store(name="pretrain_zannone2019", node=Zannone2019PretrainConfig)

# --- TRAINING METHODS ---

# shim2018


@dataclass
class Shim2018AgentConfig:
    eps_steps: int
    replay_buffer_size: int
    replay_buffer_batch_size: int
    num_optim: int
    init_random_frames: int  # how many frames to collect before starting training
    eps_init: float
    eps_end: float
    lr: float
    update_tau: float
    max_grad_norm: float
    replay_buffer_alpha: float
    replay_buffer_beta_init: float
    replay_buffer_beta_end: float
    delay_value: bool
    double_dqn: bool
    action_value_num_cells: list[int]
    action_value_dropout: float


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

    device: str
    seed: int
    pretrained_model_lr: float
    activate_joint_training_after_n_batches: int
    output_artifact_aliases: list[str]
    evaluate_final_performance: bool
    eval_only_n_samples: int | None


cs.store(name="train_shim2018", node=Shim2018TrainConfig)

# randomdummy


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
class Ma2018PretraingConfig:
    dataset_artifact_name: str
    output_artifact_aliases: list[str] = field(default_factory=lambda: [])

    batch_size: int = 128
    seed: int = 42
    device: str = "cuda"

    pointnet: Ma2018PointNetConfig = field(default_factory=Ma2018PointNetConfig)
    partial_vae: Ma2018PartialVAEConfig = field(default_factory=Ma2018PartialVAEConfig)
    classifier: Ma2018ClassifierConfig = field(default_factory=Ma2018ClassifierConfig)


cs.store(name="pretrain_ma2018", node=Ma2018PretraingConfig)


@dataclass
class Ma2018TraingConfig:
    pretrained_model_artifact_name: str
    hard_budget: int
    device: str = "cuda"
    seed: int = 42
    output_artifact_aliases: list[str] = field(default_factory=lambda: [])


cs.store(name="train_ma2018", node=Ma2018TraingConfig)


@dataclass
class RandomDummyTrainConfig:
    dataset_artifact_name: str
    hard_budget: int  # not used, but pretend that it is
    seed: int
    output_artifact_aliases: list[str]


# zannone2019


@dataclass
class Zannone2019AgentConfig:
    gamma: float
    lmbda: float
    clip_epsilon: float
    entropy_bonus: bool
    entropy_coef: float
    critic_coef: float
    loss_critic_type: str
    lr: float
    max_grad_norm: float
    sub_batch_size: int
    num_epochs: int
    value_num_cells: list[int]
    value_dropout: float
    policy_num_cells: list[int]
    policy_dropout: float


@dataclass
class Zannone2019TrainConfig:
    pretrained_model_artifact_name: str
    n_agents: int
    hard_budget: int
    agent: Zannone2019AgentConfig
    n_batches: int  # how many batches to train the agent
    batch_size: int  # batch size for collector
    eval_every_n_batches: int  # how often to evaluate the agent
    eval_max_steps: int  # maximum allowed number of steps in an evaluation episode
    n_eval_episodes: int  # how many episodes to average over in evaluation
    n_generated_samples: (
        int  # how many artificial samples to generate using pretrained model
    )
    generation_batch_size: int  # which batch size to use for artificial data generation

    device: str
    seed: int
    output_artifact_aliases: list[str]
    evaluate_final_performance: bool
    eval_only_n_samples: int | None


cs.store(name="train_randomdummy", node=RandomDummyTrainConfig)

# --- TRAINING CLASSIFIERS ---


@dataclass
class TrainMaskedMLPClassifierConfig:
    dataset_artifact_name: str
    batch_size: int
    epochs: int
    limit_train_batches: int | None
    limit_val_batches: int | None
    min_masking_probability: float
    max_masking_probability: float
    eval_only_n_samples: int | None  # if specified, only evaluate on this many samples
    lr: float
    seed: int
    device: str
    num_cells: list[int]
    dropout: float
    output_artifact_aliases: list[str]
    evaluate_final_performance: bool


cs.store(name="train_masked_mlp_classifier", node=TrainMaskedMLPClassifierConfig)

# --- EVALUATION ---


@dataclass
class EvalConfig:
    trained_method_artifact_name: str
    trained_classifier_artifact_name: str | None  # if None, use the method's classifier
    seed: int
    device: str
    output_artifact_aliases: list[str]
    eval_only_n_samples: int | None  # if specified, only evaluate on this many samples
    batch_size: int


cs.store(name="eval", node=EvalConfig)

# --- PLOTTING ---


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
