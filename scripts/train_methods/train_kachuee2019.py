import gc
import logging
from pathlib import Path
from typing import Any, cast
from tempfile import TemporaryDirectory

import hydra
from omegaconf import OmegaConf
import torch
from torch import optim
from torch.nn import functional as F
from torchrl.collectors import SyncDataCollector
from torchrl.envs import ExplorationType, check_env_specs, set_exploration_type
from afa_rl.agents import Agent
from tqdm import tqdm

import wandb
from afa_rl.afa_env import AFAEnv
from afa_rl.afa_methods import RLAFAMethod
from afa_rl.kachuee2019.agents import Kachuee2019Agent
from afa_rl.kachuee2019.models import (
    Kachuee2019AFAClassifier,
    Kachuee2019AFAPredictFn,
    Kachuee2019PQModule,
)
from afa_rl.kachuee2019.reward import get_kachuee2019_reward_fn
from afa_rl.datasets import get_afa_dataset_fn
from afa_rl.utils import (
    get_eval_metrics,
    module_norm,
)
from common.afa_methods import RandomDummyAFAMethod
from common.config_classes import (
    Kachuee2019TrainConfig,
)
from common.utils import get_class_probabilities, load_dataset_artifact, set_seed

from eval.metrics import eval_afa_method
from eval.utils import plot_metrics


log = logging.getLogger(__name__)


@hydra.main(
    version_base=None, config_path="../../conf/train/kachuee2019", config_name="config"
)
def main(cfg: Kachuee2019TrainConfig):
    log.debug(cfg)
    set_seed(cfg.seed)
    torch.set_float32_matmul_precision("medium")
    device = torch.device(cfg.device)

    run = wandb.init(
        config=cast(dict[str, Any], OmegaConf.to_container(cfg, resolve=True)),
        job_type="training",
        tags=["kachuee2019"],
    )

    # Log W&B run URL
    log.info(f"W&B run initialized: {run.name} ({run.id})")
    log.info(f"W&B run URL: {run.url}")

    # Load dataset artifact
    train_dataset, val_dataset, _, dataset_metadata = load_dataset_artifact(
        cfg.dataset_artifact_name
    )
    train_class_probabilities = get_class_probabilities(train_dataset.labels)
    log.debug(f"Class probabilities in training set: {train_class_probabilities}")
    class_weights = F.softmax(1 / train_class_probabilities, dim=-1).to(device)
    n_features = train_dataset.features.shape[-1]
    n_classes = train_dataset.labels.shape[-1]

    pq_module = Kachuee2019PQModule(
        n_features=n_features, n_classes=n_classes, cfg=cfg.pq_module
    ).to(device)
    pq_module_optim = optim.Adam(pq_module.parameters(), lr=cfg.predictor_lr)

    reward_fn = get_kachuee2019_reward_fn(
        pq_module=pq_module, method=cfg.reward_method, mcdrop_samples=cfg.mcdrop_samples
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

    agent: Agent = Kachuee2019Agent(
        action_spec=train_env.action_spec,
        action_mask_key="action_mask",
        module_device=torch.device(cfg.device),
        replay_buffer_device=torch.device(cfg.device),
        pq_module=pq_module,
        cfg=cfg.agent,
    )

    collector = SyncDataCollector(
        train_env,
        agent.get_policy(),
        frames_per_batch=cfg.batch_size,
        total_frames=cfg.n_batches * cfg.batch_size,
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

            # Train predictor
            class_logits_next, _qvalues_next = pq_module(td["next", "masked_features"])
            class_loss_next = F.cross_entropy(
                class_logits_next, td["next", "label"], weight=class_weights
            ).mean()
            pq_module_optim.zero_grad()
            class_loss_next.backward()
            pq_module_optim.step()

            # Log training info
            run.log(
                {
                    f"train/{k}": v
                    for k, v in (
                        loss_info
                        | agent.get_cheap_info()
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
                        | {"class_loss": class_loss_next.cpu().item()}
                    ).items()
                },
            )

            if batch_idx != 0 and batch_idx % cfg.eval_every_n_batches == 0:
                with (
                    torch.no_grad(),
                    set_exploration_type(ExplorationType.DETERMINISTIC),
                ):
                    # HACK: Set the action spec of the agent to the eval env action spec
                    agent.egreedy_tdmodule._spec = eval_env.action_spec  # pyright: ignore
                    td_evals = [
                        eval_env.rollout(
                            cfg.eval_max_steps, agent.get_policy()
                        ).squeeze(0)
                        for _ in tqdm(range(cfg.n_eval_episodes), desc="Evaluating")
                    ]
                    # Reset the action spec of the agent to the train env action spec
                    agent.egreedy_tdmodule._spec = train_env.action_spec  # pyright: ignore
                metrics_eval = get_eval_metrics(
                    td_evals, Kachuee2019AFAPredictFn(pq_module)
                )
                run.log(
                    {
                        **{
                            f"eval/{k}": v
                            for k, v in (
                                metrics_eval
                                | agent.get_expensive_info()
                                | {
                                    "p_net_norm": module_norm(pq_module.layers_p),
                                    "q_net_norm": module_norm(pq_module.layers_q),
                                }
                            ).items()
                        },
                    }
                )

    except KeyboardInterrupt:
        pass
    finally:
        afa_method = RLAFAMethod(
            agent.get_policy().to("cpu"),
            Kachuee2019AFAClassifier(pq_module, device=torch.device("cpu")),
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
                fig_metrics = plot_metrics(metrics)
                # Also check performance of dummy method for comparison
                dummy_afa_method = RandomDummyAFAMethod(
                    device=torch.device("cpu"), n_classes=n_classes
                )
                dummy_metrics = eval_afa_method(
                    dummy_afa_method.select,
                    val_dataset,
                    cfg.hard_budget,
                    afa_method.predict,
                    only_n_samples=cfg.eval_only_n_samples,
                )
                fig_dummy_metrics = plot_metrics(dummy_metrics)
                run.log(
                    {
                        "final_performance_plot": fig_metrics,
                        "final_dummy_performance_plot": fig_dummy_metrics,
                    }
                )

            # Save the model as a WandB artifact
            # Save the name of the afa method class as metadata
            afa_method_artifact = wandb.Artifact(
                name=f"train_kachuee2019-{cfg.dataset_artifact_name.split(':')[0]}-budget_{cfg.hard_budget}-seed_{cfg.seed}",
                type="trained_method",
                metadata={
                    "method_type": "kachuee2019",
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
