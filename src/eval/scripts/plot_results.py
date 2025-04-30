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
from torch import Tensor
import matplotlib.pyplot as plt

from common.registry import AFA_DATASET_REGISTRY, AFA_METHOD_REGISTRY

from eval.utils import get_eval_results_with_fixed_keys

AFA_METHOD_TYPES = list(AFA_METHOD_REGISTRY.keys())
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

def tensor_list_to_stats(tensor_list: List[Tensor]) -> tuple[Tensor, Tensor]:
    """Stack a list of 1‑D tensors → compute mean & std along axis 0."""
    stacked = torch.stack(tensor_list)  # (N, B)
    mean = stacked.mean( 0)
    std = stacked.std(0)
    return mean, std


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------------

def main() -> None:
    for dataset_name in AFA_DATASET_REGISTRY.keys():
        print(f"Processing dataset: {dataset_name}")
        for is_builtin_classifier in (True, False):
            print(" Processing builtin classifier:", is_builtin_classifier)
            # Collect stats per method → {metric: {method: (mean, std)}}
            metric_table: Dict[str, Dict[str, tuple[Tensor, Tensor]]] = {
                "accuracy": {},
                "f1": {},
            }

            at_least_one_method = False
            for method_type in AFA_METHOD_TYPES:
                print("    Processing method:", method_type)
                # ------------------------------------------------------------------
                # Query results
                # ------------------------------------------------------------------

                results = get_eval_results_with_fixed_keys(  # type: ignore
                    fixed_params_mapping={
                        "dataset_type": dataset_name,
                        "method_type": method_type,
                        "method_hard_budget": BUDGET,
                        "is_builtin_classifier": is_builtin_classifier,
                        "eval_hard_budget": BUDGET,
                    }
                )

                if results:
                    print(f"      [INFO] Found results for {dataset_name} | builtin_clf={is_builtin_classifier} | {method_type}")
                else:
                    print(f"      [WARN] No results for {dataset_name} | builtin_clf={is_builtin_classifier} | {method_type}")
                    continue

                # Produce a plot for this dataset/method combination
                at_least_one_method = True

                # ------------------------------------------------------------------
                # Aggregate accuracy_all and f1_all across result dicts
                # ------------------------------------------------------------------
                acc_tensors = [d["accuracy_all"] for d in results]
                f1_tensors = [d["f1_all"] for d in results]

                acc_mean, acc_std = tensor_list_to_stats(acc_tensors)
                f1_mean, f1_std = tensor_list_to_stats(f1_tensors)

                metric_table["accuracy"][method_type] = (acc_mean, acc_std)
                metric_table["f1"][method_type] = (f1_mean, f1_std)

            if not at_least_one_method:
                print(f"  [WARN] No methods for {dataset_name} | builtin_clf={is_builtin_classifier}")
                continue
            # ----------------------------------------------------------------------
            # Render plots for both metrics
            # ----------------------------------------------------------------------
            x = np.arange(1, len(next(iter(metric_table["accuracy"].values()))[0]) + 1)

            for metric in ("accuracy", "f1"):
                save_dir = PLOTS_DIR / dataset_name / metric / ("0" if is_builtin_classifier else "1")
                ensure_dir(save_dir)

                plt.figure(figsize=(6, 4))
                for method_type, (mean_vec, std_vec) in metric_table[metric].items():
                    plt.errorbar(
                        x,
                        mean_vec,
                        yerr=std_vec,
                        marker="o",
                        capsize=3,
                        label=method_type,
                    )

                plt.title(f"{metric.upper()} | {dataset_name} | builtin_clf={is_builtin_classifier}")
                plt.xlabel("Feature‑selection budget (B)")
                plt.ylabel(metric.upper())
                plt.legend()
                plt.tight_layout()

                outfile = save_dir / f"{metric}_{dataset_name}_{int(not is_builtin_classifier)}.png"
                plt.savefig(outfile, dpi=300)
                plt.close()
                print(f"[√] Saved → {outfile}")


if __name__ == "__main__":
    main()
