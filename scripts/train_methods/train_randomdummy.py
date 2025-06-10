from functools import partial
import gc
import logging
from pathlib import Path
from typing import Any, cast
from tempfile import TemporaryDirectory

import hydra
from omegaconf import OmegaConf
import torch
from tensordict import TensorDictBase
from torch import optim
from torch.nn import functional as F
from torchrl.collectors import SyncDataCollector
from torchrl.envs import ExplorationType, check_env_specs, set_exploration_type
from afa_rl.agents import Agent
from tqdm import tqdm
from dacite import from_dict

import wandb
from afa_rl.afa_env import AFAEnv, get_common_reward_fn
from afa_rl.afa_methods import RLAFAMethod
from afa_rl.shim2018.agents import Shim2018Agent
from afa_rl.datasets import get_afa_dataset_fn
from afa_rl.shim2018.models import (
    LitShim2018EmbedderClassifier,
    Shim2018AFAClassifier,
    Shim2018AFAPredictFn,
)
from afa_rl.shim2018.models import (
    get_shim2018_model_from_config,
)
from afa_rl.utils import (
    afacontext_optimal_selection,
    module_norm,
)
from common.afa_methods import RandomDummyAFAMethod
from common.config_classes import (
    RandomDummyTrainConfig,
    Shim2018PretrainConfig,
    Shim2018TrainConfig,
)
from common.custom_types import (
    AFADataset,
    AFAPredictFn,
)
from common.utils import get_class_probabilities, load_dataset_artifact, set_seed

from eval.metrics import eval_afa_method
from eval.utils import plot_metrics


log = logging.getLogger(__name__)


@hydra.main(
    version_base=None, config_path="../../conf/train/randomdummy", config_name="config"
)
def main(cfg: RandomDummyTrainConfig):
    log.debug(cfg)
    set_seed(cfg.seed)
    torch.set_float32_matmul_precision("medium")

    run = wandb.init(
        config=cast(dict[str, Any], OmegaConf.to_container(cfg, resolve=True)),
        job_type="training",
    )

    # Load dataset artifact
    train_dataset, val_dataset, _, dataset_metadata = load_dataset_artifact(
        cfg.dataset_artifact_name
    )

    # Get number of classes from the dataset
    n_classes = train_dataset.labels.shape[-1]

    afa_method = RandomDummyAFAMethod(device=torch.device("cpu"), n_classes=n_classes)
    # Save the method to a temporary directory and load it again to ensure it is saved correctly
    with TemporaryDirectory(delete=False) as tmp_path_str:
        tmp_path = Path(tmp_path_str)
        afa_method.save(tmp_path)

        # Save the model as a WandB artifact
        # Save the name of the afa method class as metadata
        afa_method_artifact = wandb.Artifact(
            name=f"train_randomdummy-{cfg.dataset_artifact_name.split(':')[0]}-budget_{cfg.hard_budget}-seed_{cfg.seed}",
            type="trained_method",
            metadata={
                "afa_method_class": afa_method.__class__.__name__,
                "method_type": "shim2018",
                "dataset_artifact_name": cfg.dataset_artifact_name,
                "dataset_type": dataset_metadata["dataset_type"],
                "budget": cfg.hard_budget,
                "seed": cfg.seed,
            },
        )

        afa_method_artifact.add_dir(str(tmp_path))
        run.log_artifact(afa_method_artifact, aliases=cfg.output_artifact_aliases)

    run.finish()

    gc.collect()  # Force Python GC
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # Release cached memory held by PyTorch CUDA allocator
        torch.cuda.synchronize()  # Optional, wait for CUDA ops to finish


if __name__ == "__main__":
    main()
