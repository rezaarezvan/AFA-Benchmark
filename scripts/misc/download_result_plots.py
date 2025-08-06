"""Calculate the average time required to train each method presented in a plotting run."""

import re


import logging
from pathlib import Path
import shutil

import hydra
import torch
import wandb

from common.config_classes import PlotDownloadConfig
from omegaconf import OmegaConf



def process_figure_artifact(figure_artifact, files):
    # extract the svg file saved in this artifact and save it to files
    artifact_dir = Path(figure_artifact.download())
    figure_file_path = [f for f in artifact_dir.iterdir()][0]
    files.append(figure_file_path)


def process_plot_artifact(cfg: PlotDownloadConfig, plot_run):
    files = []
    figure_artifacts = [
        artifact
        for artifact in plot_run.logged_artifacts()
        if artifact.type == "publication_figure"
    ]
    for figure_artifact in figure_artifacts:
        log.debug(f"Processing {figure_artifact.name}")
        # Check if the artifact name matches something we want
        for dataset_name, budgets, metric in zip(
            cfg.datasets, cfg.budgets, cfg.metrics, strict=False
        ):
            # If budgets is a dot, accept any budget
            if budgets.strip() == ".":
                budget = None
                if is_match(
                    figure_artifact.name,
                    dataset_name,
                    budget,
                    metric,
                    cfg.file_type,
                ):
                    log.info(f"MATCH: {figure_artifact.name}")
                    process_figure_artifact(figure_artifact, files)
            else:
                for budget in map(int, budgets.split(" ")):
                    if is_match(
                        figure_artifact.name,
                        dataset_name,
                        budget,
                        metric,
                        cfg.file_type,
                    ):
                        log.info(f"MATCH: {figure_artifact.name}")
                        process_figure_artifact(figure_artifact, files)
    return files


def is_match(
    artifact_name: str, dataset: str, budget: int | None, metric: str, file_type: str
):
    """Checks if the artifact_name matches the expected pattern for the given dataset, budget, and metric.
    Example artifact_name:
        figure-FashionMNIST_MaskedMLPClassifier_budget10_f1_all-svg:v0
    If budget is None, accept any budget value.
    """
    # log.info(
    #     f"checking if {artifact_name} matches figure-{dataset}_budget{budget}_{metric}-{file_type}:v?"
    # )
    if budget is None:
        budget_pattern = r"budget\d+"
    else:
        budget_pattern = f"budget{budget}"
    pattern = rf"^figure-{re.escape(dataset)}_.*_{budget_pattern}_{re.escape(metric)}-{file_type}:v\d+$"
    return re.match(pattern, artifact_name) is not None


log = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="../../conf/misc",
    config_name="download_plot_results",
)
def main(cfg: PlotDownloadConfig) -> None:
    log.debug(cfg)
    torch.set_float32_matmul_precision("medium")

    run = wandb.init(
        job_type="evaluation",
        config=OmegaConf.to_container(cfg, resolve=True),  # pyright: ignore
        dir="wandb",
    )

    # Log W&B run URL
    log.info(f"W&B run initialized: {run.name} ({run.id})")
    log.info(f"W&B run URL: {run.url}")

    plotting_run = wandb.Api().run(cfg.plotting_run_name)

    files = process_plot_artifact(cfg, plotting_run)

    # Move all files to the desired output folder
    output_path = Path(cfg.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    for file_path in files:
        dest_path = output_path / Path(file_path).name
        shutil.move(str(file_path), str(dest_path))
        log.info(f"Moved {file_path} to {dest_path}")

    run.finish()


if __name__ == "__main__":
    main()
