import hydra
from omegaconf import DictConfig, OmegaConf
from dataclasses import dataclass


@dataclass
class DatasetConfig:
    class_name: str  # e.g. "AFAContext"
    artifact_name: str  # wandb artifact name, e.g. "afa_context:latest"


@dataclass
class PretrainConfig:
    method: str  # e.g. "shim2018"
    dataset: DatasetConfig


@hydra.main(version_base=None, config_path="../conf", config_name="pretrain_method")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    main()
