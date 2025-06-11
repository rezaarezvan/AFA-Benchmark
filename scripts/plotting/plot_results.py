from collections import defaultdict
from omegaconf import OmegaConf
from matplotlib import pyplot as plt
import numpy as np
import logging
from pathlib import Path
from typing import Any
import hydra
import torch
import wandb

from common.config_classes import PlotConfig

log = logging.getLogger(__name__)


def load_eval_results(
    artifact_names: list[str],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Load eval results from wandb artifacts.

    The results are returned as a list of tuples, where each tuple contains:
    1. Info describing the evaluation (dataset type, method type, split, seed, classifier type)
    2. The actual metrics dictionary.
    """
    eval_results = []
    for artifact_name in artifact_names:
        eval_artifact = wandb.use_artifact(artifact_name, type="eval_results")
        log.info(f"Downloading evaluation artifact {artifact_name}")
        eval_artifact_dir = Path(eval_artifact.download())
        metrics = torch.load(eval_artifact_dir / "metrics.pt")
        info = {
            "dataset_type": eval_artifact.metadata["dataset_type"],
            "method_type": eval_artifact.metadata["method_type"],
            "budget": eval_artifact.metadata["budget"],
            "seed": eval_artifact.metadata["seed"],
            "classifier_type": eval_artifact.metadata["classifier_type"],
        }
        eval_results.append((info, metrics))
    return eval_results


@hydra.main(version_base=None, config_path="../../conf/plot", config_name="config")
def main(cfg: PlotConfig):
    log.debug(cfg)
    torch.set_float32_matmul_precision("medium")

    run = wandb.init(
        job_type="plotting",
        config=OmegaConf.to_container(cfg, resolve=True),  # pyright: ignore
    )

    eval_results = load_eval_results(cfg.eval_artifact_names)
    log.info("All evaluation result artifacts loaded.")

    dataset_types = set(info["dataset_type"] for (info, _) in eval_results)

    for dataset_type in dataset_types:
        log.info(f"Plotting results for dataset type: {dataset_type}")
        classifier_types = set(
            info["classifier_type"]
            for (info, _) in eval_results
            if info["dataset_type"] == dataset_type
        )
        for classifier_type in classifier_types:
            log.info(f"  Plotting results for classifier type: {classifier_type}")
            budgets = set(
                info["budget"]
                for (info, _) in eval_results
                if info["dataset_type"] == dataset_type
                and info["classifier_type"] == classifier_type
            )
            for budget in budgets:
                log.info(f"    Plotting results for budget: {budget}")
                # x-axis will be [1, budget]
                x = np.arange(1, budget + 1)
                # Organize by method_type
                grouped_metrics: dict[str, list[dict[str, torch.Tensor]]] = defaultdict(
                    list
                )

                for info, metrics in eval_results:
                    if (
                        info["dataset_type"] == dataset_type
                        and info["classifier_type"] == classifier_type
                        and info["budget"] == budget
                    ):
                        method_type = info["method_type"]
                        grouped_metrics[method_type].append(metrics)

                if not grouped_metrics:
                    continue

                for metric_cfg in cfg.metric_keys_and_descriptions:
                    fig, ax = plt.subplots()

                    for method_type, metrics_list in grouped_metrics.items():
                        log.info(
                            f"      Plotting results for method type: {method_type}"
                        )
                        # Shape: [num_runs, T]
                        data = torch.stack([m[metric_cfg.key] for m in metrics_list])
                        mean = data.mean(dim=0)
                        std = data.std(dim=0)

                        ax.plot(x, mean, label=method_type)
                        if metric_cfg.ylim is not None:
                            ax.set_ylim(*metric_cfg.ylim)
                        ax.fill_between(x, mean - std, mean + std, alpha=0.3)

                    ax.set_title(
                        f"{metric_cfg.key} – {dataset_type} | {classifier_type} | Budget: {budget}"
                    )
                    ax.set_xlabel("Number of features selected")
                    ax.set_ylabel(metric_cfg.description)
                    ax.legend()
                    ax.grid(True)

                    wandb.log(
                        {
                            f"{dataset_type}_{classifier_type}_budget{budget}_{metric_cfg.key}": wandb.Image(
                                fig
                            )
                        }
                    )
                    plt.close(fig)


if __name__ == "__main__":
    main()
