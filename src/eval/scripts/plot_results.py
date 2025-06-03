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


# @hydra.main(version_base=None, config_path="../../../conf/plot", config_name="tmp")
# def main(cfg: PlotConfig):
#     log.debug(cfg)
#     torch.set_float32_matmul_precision("medium")
#
#     run = wandb.init(
#         job_type="evaluation",
#         config=OmegaConf.to_container(cfg, resolve=True),  # pyright: ignore
#     )
#
#     eval_results = load_eval_results(cfg.eval_artifact_names)
#
#     # Produce one plot per (dataset_type, classifier_type, budget) combination. Each method_type gets a separate line, seed and split are averaged over.
#
#     # First get a list of all datasets that we have results for
#     dataset_types = set(info["dataset_type"] for (info, _) in eval_results)
#
#     for dataset_type in dataset_types:
#         # Get all classifier types for this dataset type
#         classifier_types = set(
#             info["classifier_type"]
#             for (info, _) in eval_results
#             if info["dataset_type"] == dataset_type
#         )
#         for classifier_type in classifier_types:
#             # Get all budgets for this dataset and classifier type
#             budgets = set(
#                 info["budget"]
#                 for (info, _) in eval_results
#                 if info["dataset_type"] == dataset_type
#                 and info["classifier_type"] == classifier_type
#             )
#             for budget in budgets:
#                 # Now get all results for this dataset, classifier type, and budget
#                 all_metrics = [
#                     metrics
#                     for (info, metrics) in eval_results
#                     if info["dataset_type"] == dataset_type
#                     and info["classifier_type"] == classifier_type
#                     and info["budget"] == budget
#                 ]
#                 # Each element in all_metrics is a dict with keys like "accuracy", "f1", etc. Create two new dicts: one that contains the average metrics and one that contains the standard deviation.
#                 avg_metrics = {}
#                 std_metrics = {}


@hydra.main(version_base=None, config_path="../../../conf/plot", config_name="tmp")
def main(cfg: PlotConfig):
    log.debug(cfg)
    torch.set_float32_matmul_precision("medium")

    run = wandb.init(
        job_type="evaluation",
        config=OmegaConf.to_container(cfg, resolve=True),  # pyright: ignore
    )

    eval_results = load_eval_results(cfg.eval_artifact_names)

    dataset_types = set(info["dataset_type"] for (info, _) in eval_results)

    for dataset_type in dataset_types:
        classifier_types = set(
            info["classifier_type"]
            for (info, _) in eval_results
            if info["dataset_type"] == dataset_type
        )
        for classifier_type in classifier_types:
            budgets = set(
                info["budget"]
                for (info, _) in eval_results
                if info["dataset_type"] == dataset_type
                and info["classifier_type"] == classifier_type
            )
            for budget in budgets:
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

                # Extract all metric keys
                all_metric_keys = list(next(iter(grouped_metrics.values()))[0].keys())

                for metric_key in all_metric_keys:
                    fig, ax = plt.subplots()

                    for method_type, metrics_list in grouped_metrics.items():
                        # Shape: [num_runs, T]
                        data = np.stack([np.array(m[metric_key]) for m in metrics_list])
                        mean = data.mean(axis=0)
                        std = data.std(axis=0)
                        x = np.arange(len(mean))

                        ax.plot(x, mean, label=method_type)
                        ax.fill_between(x, mean - std, mean + std, alpha=0.3)

                    ax.set_title(
                        f"{metric_key} – {dataset_type} | {classifier_type} | Budget: {budget}"
                    )
                    ax.set_xlabel("Step")
                    ax.set_ylabel(metric_key)
                    ax.legend()
                    ax.grid(True)

                    wandb.log(
                        {
                            f"{dataset_type}_{classifier_type}_budget{budget}_{metric_key}": wandb.Image(
                                fig
                            )
                        }
                    )
                    plt.close(fig)


if __name__ == "__main__":
    main()
