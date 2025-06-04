from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore


# @dataclass
# class ArtifactConfig:
#     name: str  # e.g "pretrain_shim2018-cube_split_1:May26"


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

    device: str = "cuda"
    seed: int = 42
    lr: float = 1e-3
    max_masking_probability: float = 0.9
    encoder: Shim2018EncoderConfig = field(default_factory=Shim2018EncoderConfig)
    classifier: Shim2018ClassifierConfig = field(
        default_factory=Shim2018ClassifierConfig
    )
    output_artifact_aliases: list[str] = field(default_factory=lambda: [])


cs = ConfigStore.instance()
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


cs.store(name="train_shim2018", node=Shim2018TrainConfig)


@dataclass
class TrainMaskedMLPClassifierConfig:
    dataset_artifact_name: str
    batch_size: int
    epochs: int
    max_masking_probability: float = 0.9
    lr: float = 1e-3
    seed: int = 42
    device: str = "cuda"
    num_cells: list[int] = field(default_factory=lambda: [128, 128])
    dropout: float = 0.1
    output_artifact_aliases: list[str] = field(default_factory=lambda: [])


cs.store(name="train_masked_mlp_classifier", node=TrainMaskedMLPClassifierConfig)


@dataclass
class EvalConfig:
    trained_method_artifact_name: str
    trained_classifier_artifact_name: str | None  # if None, use the method's classifier
    seed: int = 42
    output_artifact_aliases: list[str] = field(default_factory=lambda: [])


@dataclass
class PlotConfig:
    eval_artifact_names: list[str]
    metric_keys_and_descriptions: list[
        list[str]
    ]  # Inner list has two elements: [metric_key, description]


cs.store(name="plot", node=PlotConfig)
