
"""
Aggregate evaluation results and create comparison plots.

Assumptions
-----------
* Each *.pt file produced by `evaluator` is a plain Python dict with
    ├─ "accuracy"               : float
    ├─ "accuracy_all"           : list[float]
    └─ "num_features_selected"  : list[int]

* The registry below is the single source of truth for where those
  *.pt files live on disk.

* torch, numpy and matplotlib are installed.
"""

from pathlib import Path
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from common.registry import EVALUATION_REGISTRY

# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------
def _load_result(path: str) -> dict:
    """Load a single *.pt result file."""
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return torch.load(path, map_location="cpu")


def _aggregate_accuracy_curves(curves: list[list[float]]):
    """
    Stack variable-length accuracy curves.

    Returns
    -------
    xs          : np.ndarray, shape (L,)
    mean_curve  : np.ndarray, shape (L,)
    std_curve   : np.ndarray, shape (L,)
        where L is the longest curve length.
    """
    max_len = max(len(c) for c in curves)
    means, stds = [], []
    for i in range(max_len):
        vals = [c[i] for c in curves if i < len(c)]
        means.append(np.mean(vals))
        stds.append(np.std(vals, ddof=0))       # population std (len==1 → 0.0)
    xs = np.arange(max_len)
    return xs, np.asarray(means), np.asarray(stds)


def _aggregate_scalar(values: list[float]):
    """Return mean and (population) std as a tuple."""
    arr = np.asarray(values)
    return arr.mean(), arr.std(ddof=0)


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------
def _plot_dataset(dataset_name: str, methods_dict: dict[str, list[str]]):
    """
    Create a single figure for one dataset, containing
      • a per-step accuracy curve (with vertical error bars) for every method
      • one summary point per method with horizontal & vertical error bars
    """
    fig, ax = plt.subplots()

    for method, split_files in methods_dict.items():
        # ------------------------------------------------------------------
        # Load all splits for the current method
        # ------------------------------------------------------------------
        curves, finals, feat_all = [], [], []
        for f in split_files:
            res = _load_result(f)
            curves.append(res["accuracy_all"])
            finals.append(res["accuracy"])
            feat_all.extend(res["num_features_selected"])

        # ------------------------------------------------------------------
        # Line: per-step accuracy ± std
        # ------------------------------------------------------------------
        xs, means, stds = _aggregate_accuracy_curves(curves)
        line = ax.errorbar(
            xs,
            means,
            yerr=stds,
            label=method,
            linewidth=1.6,
            capsize=3,
        )

        # ------------------------------------------------------------------
        # Summary point: <features, accuracy>
        # ------------------------------------------------------------------
        x_mu, x_std = _aggregate_scalar(feat_all)
        y_mu, y_std = _aggregate_scalar(finals)

        ax.errorbar(
            x_mu,
            y_mu,
            xerr=x_std,
            yerr=y_std,
            fmt="o",
            capsize=3,
            color=line.lines[0].get_color(),  # match curve colour
        )

    # ------------------------------------------------------------------
    # Final figure styling
    # ------------------------------------------------------------------
    ax.set_xlabel("Number of features selected")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Method comparison on '{dataset_name}'")
    ax.grid(alpha=0.3)
    ax.legend()

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    out_dir = Path(f"results/plots/{dataset_name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{dataset_name}_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[✓] Saved plot → {out_dir}/{dataset_name}_comparison.png")


# ---------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------
def main():
    for dataset_name, ds_cfg in EVALUATION_REGISTRY["datasets"].items():
        _plot_dataset(dataset_name, ds_cfg["methods"])


if __name__ == "__main__":
    main()
