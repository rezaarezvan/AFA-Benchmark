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
from common.config_classes import Shim2018PretrainConfig, Shim2018TrainConfig
from common.custom_types import (
    AFADataset,
    AFAPredictFn,
)
from common.utils import get_class_probabilities, load_dataset_artifact, set_seed

from eval.metrics import eval_afa_method
from eval.utils import plot_metrics


def get_eval_metrics(
    eval_tds: list[TensorDictBase], afa_predict_fn: AFAPredictFn
) -> dict[str, Any]:
    eval_metrics = {}
    eval_metrics["reward_sum"] = 0.0
    eval_metrics["actions"] = []
    n_correct_samples = 0
    for td in eval_tds:
        eval_metrics["reward_sum"] += td["next", "reward"].sum()
        eval_metrics["actions"].extend(td["action"].tolist())
        # Check whether prediction is correct
        td_end = td[-1:]
        probs = afa_predict_fn(
            td_end["next", "masked_features"], td_end["next", "feature_mask"]
        )
        if probs.argmax(dim=-1) == td_end["label"].argmax(dim=-1):
            n_correct_samples += 1
    eval_metrics["reward_sum"] /= len(eval_tds)
    eval_metrics["actions"] = wandb.Histogram(eval_metrics["actions"], num_bins=20)
    eval_metrics["accuracy"] = n_correct_samples / len(eval_tds)
    return eval_metrics


def afacontext_benchmark_policy(
    tensordict: TensorDictBase,
) -> TensorDictBase:
    """Select features optimally in AFAContext."""
    masked_features = tensordict["masked_features"]
    feature_mask = tensordict["feature_mask"]

    selection = afacontext_optimal_selection(masked_features, feature_mask)

    tensordict["action"] = selection
    return tensordict


def load_pretrained_model_artifacts(
    artifact_name: str,
) -> tuple[
    AFADataset,  # train dataset
    AFADataset,  # val dataset
    AFADataset,  # test dataset
    dict[str, Any],  # dataset metadata
    LitShim2018EmbedderClassifier,
    Shim2018PretrainConfig,
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
    pretrained_model_config: Shim2018PretrainConfig = from_dict(
        data_class=Shim2018PretrainConfig, data=pretrained_model_config_dict
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

    pretrained_model = get_shim2018_model_from_config(
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
    version_base=None, config_path="../../conf/train/shim2018", config_name="config"
)
def main(cfg: Shim2018TrainConfig):
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
    train_class_probabilities = get_class_probabilities(train_dataset.labels)
    log.debug(f"Class probabilities in training set: {train_class_probabilities}")
    class_weights = F.softmax(1 / train_class_probabilities, dim=-1).to(device)
    n_features = train_dataset.features.shape[-1]
    n_classes = train_dataset.labels.shape[-1]

    pretrained_model.eval()
    pretrained_model = pretrained_model.to(device)

    pretrained_model_optim = optim.Adam(
        pretrained_model.parameters(), lr=cfg.pretrained_model_lr
    )

    # The RL reward function depends on a specific AFAClassifier
    reward_fn = get_common_reward_fn(
        Shim2018AFAPredictFn(pretrained_model),
        loss_fn=partial(F.cross_entropy, weight=class_weights),
    )

    # MDP expects special dataset functions
    train_dataset_fn = get_afa_dataset_fn(train_dataset.features, train_dataset.labels)
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

    agent: Agent = Shim2018Agent(
        embedder=pretrained_model.embedder,
        embedding_size=pretrained_model_config.encoder.output_size,
        n_features=n_features,
        action_mask_key="action_mask",
        action_spec=train_env.action_spec,
        _device=device,
        gamma=1.0,
        loss_function="l2",
        replay_buffer_device=device,
        **OmegaConf.to_container(cfg.agent, resolve=True),  # pyright: ignore
    )

    # Manual debugging
    td = train_env.reset()
    td = agent.policy(td)
    td = train_env.step(td)

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

            # Train classifier and embedder jointly if we have reached the correct batch
            if batch_idx >= cfg.activate_joint_training_after_n_batches:
                pretrained_model.train()
                pretrained_model_optim.zero_grad()

                _, logits_next = pretrained_model(
                    td["next", "masked_features"], td["next", "feature_mask"]
                )
                class_loss_next = F.cross_entropy(
                    logits_next, td["next", "label"], weight=class_weights
                )
                class_loss_next.mean().backward()

                pretrained_model_optim.step()
                pretrained_model.eval()
            else:
                class_loss_next = torch.zeros((1,), device=device, dtype=torch.float32)

            # Log training info
            run.log(
                {
                    f"train/{k}": v
                    for k, v in (
                        loss_info
                        | agent.get_train_info()
                        | {
                            "reward": td["next", "reward"].mean().item(),
                            # "action value": td["action_value"].mean().item(),
                            "chosen action value": td["chosen_action_value"]
                            .mean()
                            .item(),
                            # "actions": wandb.Histogram(
                            #     td["action"].tolist(), num_bins=20
                            # ),
                        }
                        | {"class_loss": class_loss_next.mean().cpu().item()}
                    ).items()
                },
            )

            if batch_idx != 0 and batch_idx % cfg.eval_every_n_batches == 0:
                with (
                    torch.no_grad(),
                    set_exploration_type(ExplorationType.DETERMINISTIC),
                ):
                    # HACK: Set the action spec of the agent to the eval env action spec
                    agent.egreedy_module._spec = eval_env.action_spec  # pyright: ignore
                    td_evals = [
                        eval_env.rollout(cfg.eval_max_steps, agent.policy).squeeze(0)
                        for _ in tqdm(range(cfg.n_eval_episodes), desc="Evaluating")
                    ]
                    benchmark_td_evals = [
                        eval_env.rollout(
                            cfg.eval_max_steps, afacontext_benchmark_policy
                        ).squeeze(0)
                        for _ in tqdm(range(cfg.n_eval_episodes), desc="Evaluating")
                    ]
                    # Reset the action spec of the agent to the train env action spec
                    agent.egreedy_module._spec = train_env.action_spec  # pyright: ignore
                metrics_eval = get_eval_metrics(
                    td_evals, Shim2018AFAPredictFn(pretrained_model)
                )
                benchmark_metrics_eval = get_eval_metrics(
                    benchmark_td_evals,
                    Shim2018AFAPredictFn(pretrained_model),
                )
                run.log(
                    {
                        **{
                            f"eval/{k}": v
                            for k, v in (
                                metrics_eval
                                | agent.get_eval_info()
                                | {
                                    "classifier_norm": module_norm(
                                        pretrained_model.classifier
                                    ),
                                    "embedder_norm": module_norm(
                                        pretrained_model.embedder
                                    ),
                                }
                            ).items()
                        },
                        **{
                            f"benchmark_eval/{k}": v
                            for k, v in benchmark_metrics_eval.items()
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
            Shim2018AFAClassifier(pretrained_model, device=torch.device("cpu")),
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
                name=f"train_shim2018-{pretrained_model_config.dataset_artifact_name.split(':')[0]}-budget_{cfg.hard_budget}-seed_{cfg.seed}",
                type="trained_method",
                metadata={
                    "afa_method_class": afa_method.__class__.__name__,
                    "method_type": "shim2018",
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
