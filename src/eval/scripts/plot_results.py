#!/usr/bin/env python
"""Generate accuracy and F1 error‑bar plots for every dataset, AFA method and
`is_common_classifier` setting.

Expected globals (imported elsewhere before running this script):
    * AFA_DATASET_REGISTRY : dict[str, DatasetClass]
    * get_eval_results_with_fixed_keys : callable
    * (optionally) AFA_METHOD_REGISTRY  : dict[str, ...]

The script will create PNG files under:
    plots/<dataset_name>/<metric>/<is_common_classifier>/<metric>_<dataset>_<is_common_classifier>.png

Each line in a plot corresponds to an AFA method; mean and ±1 σ error bars
are computed across all result dictionaries returned by
`get_eval_results_with_fixed_keys()`.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import matplotlib.pyplot as plt

from common.registry import AFA_DATASET_REGISTRY, AFA_METHOD_REGISTRY

AFA_METHOD_NAMES = list(AFA_METHOD_REGISTRY.keys())
AFA_DATASET_NAMES = list(AFA_DATASET_REGISTRY.keys())

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Budget used when querying results
BUDGET = 10

PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def tensor_list_to_stats(vec_list: List[torch.Tensor]) -> tuple[np.ndarray, np.ndarray]:
    """Stack a list of 1‑D tensors → compute mean & std along axis 0."""
    stacked = torch.stack(vec_list)  # (N, B)
    mean = stacked.mean(dim=0).cpu().numpy()
    std = stacked.std(dim=0).cpu().numpy()
    return mean, std


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------------

def main() -> None:
    for dataset_name in AFA_DATASET_REGISTRY.keys():
        for is_common_classifier in (True, False):
            # Collect stats per method → {metric: {method: (mean, std)}}
            metric_table: Dict[str, Dict[str, tuple[np.ndarray, np.ndarray]]] = {
                "accuracy": {},
                "f1": {},
            }

            for method in AFA_METHOD_NAMES:
                # ------------------------------------------------------------------
                # Query results
                # ------------------------------------------------------------------
                train_params = {
                    "dataset_type": dataset_name,
                    "budget": BUDGET,
                }
                eval_params = {
                    "is_common_classifier": is_common_classifier,
                    "budget": BUDGET,
                }

                results = get_eval_results_with_fixed_keys(  # type: ignore
                    afa_method_name=method,
                    fixed_train_params_mapping=train_params,
                    fixed_eval_params_mapping=eval_params,
                )

                if not results:
                    print(f"[WARN] No results for {dataset_name} | {method} | common_clf={is_common_classifier}")
                    continue

                # ------------------------------------------------------------------
                # Aggregate accuracy_all and f1_all across result dicts
                # ------------------------------------------------------------------
                acc_vecs = [d["accuracy_all"].detach() for d in results]
                f1_vecs = [d["f1_all"].detach() for d in results]

                acc_mean, acc_std = tensor_list_to_stats(acc_vecs)
                f1_mean, f1_std = tensor_list_to_stats(f1_vecs)

                metric_table["accuracy"][method] = (acc_mean, acc_std)
                metric_table["f1"][method] = (f1_mean, f1_std)

            # ----------------------------------------------------------------------
            # Render plots for both metrics
            # ----------------------------------------------------------------------
            x = np.arange(1, len(next(iter(metric_table["accuracy"].values()))[0]) + 1)

            for metric in ("accuracy", "f1"):
                save_dir = PLOTS_DIR / dataset_name / metric / ("1" if is_common_classifier else "0")
                ensure_dir(save_dir)

                plt.figure(figsize=(6, 4))
                for method, (mean_vec, std_vec) in metric_table[metric].items():
                    plt.errorbar(
                        x,
                        mean_vec,
                        yerr=std_vec,
                        marker="o",
                        capsize=3,
                        label=method,
                    )

                plt.title(f"{metric.upper()} | {dataset_name} | common_clf={is_common_classifier}")
                plt.xlabel("Feature‑selection budget (B)")
                plt.ylabel(metric.upper())
                plt.legend()
                plt.tight_layout()

                outfile = save_dir / f"{metric}_{dataset_name}_{int(is_common_classifier)}.png"
                plt.savefig(outfile, dpi=300)
                plt.close()
                print(f"[√] Saved → {outfile.relative_to(Path.cwd())}")


if __name__ == "__main__":
    main()
