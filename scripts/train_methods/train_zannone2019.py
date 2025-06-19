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
from torch.nn import functional as F
from torchrl.collectors import SyncDataCollector
from torchrl.envs import ExplorationType, check_env_specs, set_exploration_type
from afa_rl.agents import Agent
from tqdm import tqdm
from dacite import from_dict

import wandb
from afa_rl.afa_env import AFAEnv, get_common_reward_fn
from afa_rl.afa_methods import RLAFAMethod
from afa_rl.zannone2019.agents import Zannone2019Agent
from afa_rl.datasets import get_afa_dataset_fn
from afa_rl.zannone2019.models import (
    Zannone2019PretrainingModel,
    Zannone2019AFAClassifier,
    Zannone2019AFAPredictFn,
)
from afa_rl.zannone2019.utils import (
    get_zannone2019_model_from_config,
)
from afa_rl.utils import (
    get_eval_metrics,
    module_norm,
)
from common.config_classes import Zannone2019PretrainConfig, Zannone2019TrainConfig
from common.custom_types import (
    AFADataset,
    AFAPredictFn,
)
from common.utils import get_class_probabilities, load_dataset_artifact, set_seed

from eval.metrics import eval_afa_method
from eval.utils import plot_metrics


def load_pretrained_model_artifacts(
    artifact_name: str,
) -> tuple[
    AFADataset,  # train dataset
    AFADataset,  # val dataset
    AFADataset,  # test dataset
    dict[str, Any],  # dataset metadata
    Zannone2019PretrainingModel,
    Zannone2019PretrainConfig,
]:
    """Load a pretrained model and the dataset it was trained on, from a WandB artifact."""
    pretrained_model_artifact = wandb.use_artifact(
        artifact_name, type="pretrained_model"
    )
    pretrained_model_artifact_dir = Path(pretrained_model_artifact.download())
    # The dataset dir should contain a file called model.pt
    artifact_filenames = [f.name for f in pretrained_model_artifact_dir.iterdir()]
    assert {"model.pt"}.issubset(artifact_filenames), (
        f"Dataset artifact must contain a model.pt file. Instead found: {artifact_filenames}"
    )

    # Access config of the run that produced this pretrained model
    pretraining_run = pretrained_model_artifact.logged_by()
    assert pretraining_run is not None, (
        "Pretrained model artifact must be logged by a run."
    )
    pretrained_model_config_dict = pretraining_run.config
    pretrained_model_config: Zannone2019PretrainConfig = from_dict(
        data_class=Zannone2019PretrainConfig, data=pretrained_model_config_dict
    )

    # Load the dataset that the pretrained model was trained on
    train_dataset, val_dataset, test_dataset, dataset_metadata = load_dataset_artifact(
        pretrained_model_config.dataset_artifact_name
    )

    # Get the number of features and classes from the dataset
    n_features = train_dataset.features.shape[-1]
    n_classes = train_dataset.labels.shape[-1]
    train_class_probabilities = get_class_probabilities(train_dataset.labels)
    log.debug(f"Class probabilities in training set: {train_class_probabilities}")

    pretrained_model = get_zannone2019_model_from_config(
        pretrained_model_config,
        n_features,
        n_classes,
        train_class_probabilities,
    )
    pretrained_model_checkpoint = torch.load(
        pretrained_model_artifact_dir / "model.pt",
        weights_only=True,
        map_location=torch.device("cpu"),
    )
    pretrained_model.load_state_dict(pretrained_model_checkpoint["state_dict"])

    return (
        train_dataset,
        val_dataset,
        test_dataset,
        dataset_metadata,
        pretrained_model,
        pretrained_model_config,
    )


log = logging.getLogger(__name__)


@hydra.main(
    version_base=None, config_path="../../conf/train/zannone2019", config_name="config"
)
def main(cfg: Zannone2019TrainConfig):
    log.debug(cfg)
    set_seed(cfg.seed)
    torch.set_float32_matmul_precision("medium")
    device = torch.device(cfg.device)

    run = wandb.init(
        config=cast(dict[str, Any], OmegaConf.to_container(cfg, resolve=True)),
        job_type="training",
    )

    # Load pretrained model and dataset from artifacts
    (
        train_dataset,
        val_dataset,
        _,
        dataset_metadata,
        pretrained_model,
        pretrained_model_config,
    ) = load_pretrained_model_artifacts(cfg.pretrained_model_artifact_name)
    pretrained_model = pretrained_model.to(device)
    pretrained_model.eval()
    pretrained_model.requires_grad_(False)  # zannone2019 does not train jointly
    train_class_probabilities = get_class_probabilities(train_dataset.labels)
    log.debug(f"Class probabilities in training set: {train_class_probabilities}")
    class_weights = F.softmax(1 / train_class_probabilities, dim=-1).to(device)
    n_features = train_dataset.features.shape[-1]
    n_classes = train_dataset.labels.shape[-1]

    pretrained_model = pretrained_model.to(device)

    # The RL reward function depends on a specific AFAClassifier
    reward_fn = get_common_reward_fn(
        Zannone2019AFAPredictFn(pretrained_model),
        loss_fn=partial(F.cross_entropy, weight=class_weights),
    )

    # Use the pretrained model to generate new artificial data
    generated_features = torch.zeros(cfg.n_generated_samples, n_features)
    generated_labels = torch.zeros(cfg.n_generated_samples, n_classes)
    n_generation_batches = cfg.n_generated_samples // cfg.generation_batch_size
    for batch_idx in range(n_generation_batches):
        generated_features_batch, generated_labels_batch = (
            pretrained_model.generate_data(
                latent_size=pretrained_model_config.partial_vae.latent_size,
                device=device,
                n_samples=cfg.generation_batch_size,
            )
        )
        generated_features[
            batch_idx * cfg.generation_batch_size : (batch_idx + 1)
            * cfg.generation_batch_size,
            :,
        ] = generated_features_batch
        generated_labels[
            batch_idx * cfg.generation_batch_size : (batch_idx + 1)
            * cfg.generation_batch_size,
            :,
        ] = generated_labels_batch

    # TODO: for MNIST, check that the generated data looks good

    # MDP expects special dataset functions
    train_dataset_fn = get_afa_dataset_fn(
        torch.cat([train_dataset.features, generated_features]),
        torch.cat([train_dataset.labels, generated_labels]),
    )
    val_dataset_fn = get_afa_dataset_fn(val_dataset.features, val_dataset.labels)

    train_env = AFAEnv(
        dataset_fn=train_dataset_fn,
        reward_fn=reward_fn,
        device=device,
        batch_size=torch.Size((cfg.n_agents,)),
        feature_size=n_features,
        n_classes=n_classes,
        hard_budget=cfg.hard_budget,
    )
    check_env_specs(train_env)

    eval_env = AFAEnv(
        dataset_fn=val_dataset_fn,
        reward_fn=reward_fn,
        device=device,
        batch_size=torch.Size((1,)),
        feature_size=n_features,
        n_classes=n_classes,
        hard_budget=cfg.hard_budget,
    )

    agent: Agent = Zannone2019Agent(
        action_spec=train_env.action_spec,
        _device=device,
        batch_size=cfg.batch_size,
        replay_buffer_device=device,
        # subclass kwargs
        pointnet=pretrained_model.partial_vae.pointnet,
        encoder=pretrained_model.partial_vae.encoder,
        latent_size=pretrained_model_config.partial_vae.latent_size,
        **OmegaConf.to_container(cfg.agent, resolve=True),  # pyright: ignore
    )

    collector = SyncDataCollector(
        train_env,
        agent.policy,
        frames_per_batch=cfg.batch_size,
        total_frames=cfg.n_batches * cfg.batch_size,
        # device=device,
    )

    # Training loop
    try:
        for batch_idx, tds in tqdm(
            enumerate(collector), total=cfg.n_batches, desc="Training agent..."
        ):
            collector.update_policy_weights_()

            # Collapse agent and batch dimensions
            td = tds.flatten(start_dim=0, end_dim=1)
            loss_info = agent.process_batch(td)

            # Log training info
            run.log(
                {
                    f"train/{k}": v
                    for k, v in (
                        loss_info
                        | agent.get_train_info()
                        | {
                            "reward": td["next", "reward"].mean().item(),
                        }
                    ).items()
                },
            )

            if batch_idx != 0 and batch_idx % cfg.eval_every_n_batches == 0:
                with (
                    torch.no_grad(),
                    set_exploration_type(ExplorationType.DETERMINISTIC),
                ):
                    td_evals = [
                        eval_env.rollout(cfg.eval_max_steps, agent.policy).squeeze(0)
                        for _ in tqdm(range(cfg.n_eval_episodes), desc="Evaluating")
                    ]
                metrics_eval = get_eval_metrics(
                    td_evals, Zannone2019AFAPredictFn(pretrained_model)
                )
                run.log(
                    {
                        **{
                            f"eval/{k}": v
                            for k, v in (metrics_eval | agent.get_eval_info()).items()
                        },
                    }
                )

    except KeyboardInterrupt:
        pass
    finally:
        # Convert the embedder+agent to an AFAMethod and save it as a temporary file
        agent.device = torch.device("cpu")  # Move agent to CPU for saving
        pretrained_model = pretrained_model.to(torch.device("cpu"))
        afa_method = RLAFAMethod(
            agent.policy,
            Zannone2019AFAClassifier(pretrained_model, device=torch.device("cpu")),
        )
        # Save the method to a temporary directory and load it again to ensure it is saved correctly
        with TemporaryDirectory(delete=False) as tmp_path_str:
            tmp_path = Path(tmp_path_str)
            afa_method.save(tmp_path)
            del afa_method
            afa_method = RLAFAMethod.load(
                tmp_path,
                device=torch.device("cpu"),
            )
            if cfg.evaluate_final_performance:
                # Check what the final performance of the method is. Costly for large datasets.
                metrics = eval_afa_method(
                    afa_method.select,
                    val_dataset,
                    cfg.hard_budget,
                    afa_method.predict,
                    only_n_samples=cfg.eval_only_n_samples,
                )
                fig = plot_metrics(metrics)
                run.log(
                    {
                        "final_performance_plot": fig,
                    }
                )

            # Save the model as a WandB artifact
            # Save the name of the afa method class as metadata
            afa_method_artifact = wandb.Artifact(
                name=f"train_zannone2019-{pretrained_model_config.dataset_artifact_name.split(':')[0]}-budget_{cfg.hard_budget}-seed_{cfg.seed}",
                type="trained_method",
                metadata={
                    "method_type": "zannone2019",
                    "dataset_artifact_name": pretrained_model_config.dataset_artifact_name,
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
