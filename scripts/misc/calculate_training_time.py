"""Calculate the average time required to train each method presented in a plotting run."""

from collections import defaultdict
import numpy as np

from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from pathlib import Path
from tempfile import NamedTemporaryFile

import hydra
import torch
import wandb

from common.config_classes import TrainingTimeCalculationConfig
from omegaconf import OmegaConf

import asyncio


def process_eval_artifact_sync(eval_artifact, training_times):
    eval_run = eval_artifact.logged_by()
    method_artifacts = [
        artifact
        for artifact in eval_run.used_artifacts()
        if artifact.type == "trained_method"
    ]
    assert len(method_artifacts) == 1
    method_artifact = method_artifacts[0]
    method_run = method_artifact.logged_by()
    runtime = method_run.summary["_wandb"]["runtime"]
    method_type = method_artifact.metadata["method_type"]
    # Add pretraining runtime if needed
    pretraining_artifacts = [
        artifact
        for artifact in method_run.used_artifacts()
        if artifact.type == "pretrained_model"
    ]
    for pretraining_artifact in pretraining_artifacts:
        pretraining_run = pretraining_artifact.logged_by()
        runtime += pretraining_run.summary["_wandb"]["runtime"]
        print(f"{runtime}")
    training_times[method_type].append(runtime)


def process_all_eval_artifacts(plotting_run, training_times, max_workers=8):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_eval_artifact_sync, artifact, training_times)
            for artifact in plotting_run.used_artifacts()
        ]
        for future in as_completed(futures):
            future.result()


log = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="../../conf/misc",
    config_name="calculate_training_time",
)
def main(cfg: TrainingTimeCalculationConfig) -> None:
    log.debug(cfg)
    torch.set_float32_matmul_precision("medium")

    run = wandb.init(
        job_type="time_calculation",
        config=OmegaConf.to_container(cfg, resolve=True),  # pyright: ignore
    )

    # Log W&B run URL
    log.info(f"W&B run initialized: {run.name} ({run.id})")
    log.info(f"W&B run URL: {run.url}")

    # In seconds
    training_times: defaultdict[str, list[int]] = defaultdict(list)

    plotting_run = wandb.Api().run(cfg.plotting_run_name)

    process_all_eval_artifacts(
        plotting_run, training_times, max_workers=cfg.max_workers
    )

    # We also want to store the mean and std
    processed_training_times = {}
    for method_name, values in training_times.items():
        processed_training_times[method_name] = {}
        processed_training_times[method_name]["values"] = values
        # Calculate mean and std using numpy
        processed_training_times[method_name]["mean"] = float(np.mean(values))
        processed_training_times[method_name]["std"] = float(np.std(values))

    # Save results as wandb artifact
    training_time_artifact = wandb.Artifact(
        name=f"{cfg.plotting_run_name}-training_time",
        type="training_time",
        metadata={},
    )
    with NamedTemporaryFile("w", delete=False) as f:
        metrics_save_path = Path(f.name)
        torch.save(processed_training_times, metrics_save_path)
    training_time_artifact.add_file(str(metrics_save_path), name="metrics.pt")
    run.log_artifact(training_time_artifact, aliases=cfg.output_artifact_aliases)
    run.finish()


if __name__ == "__main__":
    main()
