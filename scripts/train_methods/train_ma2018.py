import torch
import gc
import logging
from dacite import from_dict
from tempfile import TemporaryDirectory
import torch.nn as nn
from pathlib import Path
from typing import Any, cast
from afa_generative.afa_methods import Ma2018AFAMethod
from common.config_classes import Ma2018PretraingConfig, Ma2018TraingConfig
from common.custom_types import AFADataset
from common.utils import dict_to_namespace, set_seed
import wandb
import hydra
from omegaconf import OmegaConf
from common.utils import load_dataset_artifact


log = logging.getLogger(__name__)


@hydra.main(
    version_base=None, config_path="../../conf/train/ma2018", config_name="config"
)
def main(cfg: Ma2018TraingConfig):
    log.debug(cfg)
    print(OmegaConf.to_yaml(cfg))
    run = wandb.init(
        config=cast(dict[str, Any], OmegaConf.to_container(cfg, resolve=True)),
        job_type="training",
    )
    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    pretrained_model_artifact = wandb.use_artifact(
        cfg.pretrained_model_artifact_name, type="pretrained_model"
    )
    pretrained_model_artifact_dir = Path(pretrained_model_artifact.download())
    artifact_filenames = [f.name for f in pretrained_model_artifact_dir.iterdir()]
    assert {"model.pt"}.issubset(artifact_filenames), (
        f"Dataset artifact must contain a model.pt file. Instead found: {artifact_filenames}"
    )
    pretraining_run = pretrained_model_artifact.logged_by()
    assert pretraining_run is not None, (
        "Pretrained model artifact must be logged by a run."
    )
    pretrained_model_config_dict = pretraining_run.config
    pretrained_model_config: Ma2018PretraingConfig = from_dict(
        data_class=Ma2018PretraingConfig, data=pretrained_model_config_dict
    )
    train_dataset, val_dataset, test_dataset, dataset_metadata = load_dataset_artifact(
        pretrained_model_config.dataset_artifact_name
    )
    afa_method: Ma2018AFAMethod = Ma2018AFAMethod.load(
        pretrained_model_artifact_dir, device=device
    )

    with TemporaryDirectory(delete=False) as tmp_path_str:
        tmp_path = Path(tmp_path_str)
        afa_method.save(tmp_path)
        del afa_method
        afa_method = Ma2018AFAMethod.load(tmp_path, device=device)
        afa_method_artifact = wandb.Artifact(
            name=f"train_ma2018-{pretrained_model_config.dataset_artifact_name.split(':')[0]}-budget_{cfg.hard_budget}-seed_{cfg.seed}",
            type="trained_method",
            metadata={
                "method_type": "ma2018",
                "dataset_artifact_name": pretrained_model_config.dataset_artifact_name,
                "dataset_type": dataset_metadata["dataset_type"],
                "budget": cfg.hard_budget,
                "seed": cfg.seed,
            },
        )
        afa_method_artifact.add_file(str(tmp_path / "model.pt"))
        run.log_artifact(afa_method_artifact, aliases=cfg.output_artifact_aliases)
    run.finish()

    gc.collect()  # Force Python GC
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # Release cached memory held by PyTorch CUDA allocator
        torch.cuda.synchronize()  # Optional, wait for CUDA ops to finish


if __name__ == "__main__":
    main()
