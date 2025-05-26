from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore


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
class DatasetArtifactConfig:
    name: str  # e.g "cube_split_1:May26"


@dataclass
class Shim2018PretrainConfig:
    dataset_artifact: DatasetArtifactConfig
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


@dataclass
class Shim2018PretrainedModelArtifact:
    name: str  # e.g "pretrain_shim2018-cube_split_1:May26"


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


@dataclass
class Shim2018TrainConfig:
    pretrained_model_artifact: Shim2018PretrainedModelArtifact
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


cs.store(name="shim2018_train", node=Shim2018TrainConfig)
