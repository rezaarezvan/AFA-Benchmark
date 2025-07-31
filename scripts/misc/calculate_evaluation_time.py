"""Calculate the average time required to evaluate each method presented in a plotting run."""

import numpy as np

from collections import defaultdict

from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from pathlib import Path
from tempfile import NamedTemporaryFile

import hydra
import torch
import wandb

from common.config_classes import EvaluationTimeCalculationConfig
from omegaconf import OmegaConf

import asyncio


def process_eval_artifact_sync(eval_artifact, training_times):
    eval_run = eval_artifact.logged_by()
    runtime = eval_run.summary["_wandb"]["runtime"]
    method_type = eval_artifact.metadata["method_type"]
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
    config_name="calculate_evaluation_time",
)
def main(cfg: EvaluationTimeCalculationConfig) -> None:
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
    evaluation_times: defaultdict[str, list[int]] = defaultdict(list)

    plotting_run = wandb.Api().run(cfg.plotting_run_name)

    process_all_eval_artifacts(
        plotting_run, evaluation_times, max_workers=cfg.max_workers
    )

    # We also want to store the mean and std
    processed_evaluation_times = {}
    for method_name, values in evaluation_times.items():
        processed_evaluation_times[method_name] = {}
        processed_evaluation_times[method_name]["values"] = values
        # Calculate mean and std using numpy
        processed_evaluation_times[method_name]["mean"] = float(np.mean(values))
        processed_evaluation_times[method_name]["std"] = float(np.std(values))

    # Save results as wandb artifact
    evaluation_time_artifact = wandb.Artifact(
        name=f"{cfg.plotting_run_name}-evaluation_time",
        type="evaluation_time",
        metadata={},
    )
    with NamedTemporaryFile("w", delete=False) as f:
        metrics_save_path = Path(f.name)
        torch.save(processed_evaluation_times, metrics_save_path)
    evaluation_time_artifact.add_file(str(metrics_save_path), name="metrics.pt")
    run.log_artifact(evaluation_time_artifact, aliases=cfg.output_artifact_aliases)
    run.finish()


if __name__ == "__main__":
    main()
