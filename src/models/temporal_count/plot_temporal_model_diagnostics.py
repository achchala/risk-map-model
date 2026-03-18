#!/usr/bin/env python3
"""
Plot predicted vs actual crash count for the temporal model test set.

Loads test results from training, saves one figure to outputs/reports/.
Run from project root after training:
  python train_temporal_model.py
  python plot_temporal_model_diagnostics.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Project root
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import OUTPUTS_DIR


def main() -> None:
    diagnostics_path = OUTPUTS_DIR / "reports" / "temporal_model_test_results.npz"
    if not diagnostics_path.exists():
        print(f"Test results not found: {diagnostics_path}")
        print("Run training first: python train_temporal_model.py")
        sys.exit(1)

    print("Loading test-set results...")
    data = np.load(diagnostics_path)
    y_actual = np.asarray(data["y_test"], dtype=float)
    y_pred = np.asarray(data["y_pred"], dtype=float)
    y_pred = np.clip(y_pred, 0.0, None)

    reports_dir = OUTPUTS_DIR / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    # Hexbin: predicted (x) vs actual (y); cap at 10 for readability
    x_plot = np.clip(y_pred, 0, 10)
    y_plot = np.clip(y_actual, 0, 10)
    hb = ax.hexbin(x_plot, y_plot, gridsize=40, mincnt=1, cmap="viridis", edgecolors="none")
    ax.plot([0, 10], [0, 10], "r--", lw=2, label="Perfect prediction")
    # Binned mean actual by predicted
    bins = np.percentile(y_pred, np.linspace(0, 100, 21))
    bins = np.unique(bins)
    if len(bins) >= 2:
        bin_centers = (bins[:-1] + bins[1:]) / 2
        mean_actual = np.array(
            [y_actual[(y_pred >= bins[i]) & (y_pred < bins[i + 1])].mean() for i in range(len(bins) - 1)]
        )
        ax.plot(bin_centers, mean_actual, "r-", lw=2, label="Binned mean(actual)")
    ax.set_xlabel("Predicted crash count (λ per segment-week)")
    ax.set_ylabel("Actual crash count")
    ax.set_title("Predicted vs actual crash count (test set)")
    ax.legend(loc="upper right")
    ax.set_xlim(0, min(10, max(0.01, y_pred.max() * 1.02)))
    ax.set_ylim(0, min(10, max(0.01, y_actual.max() * 1.02)))
    plt.colorbar(hb, ax=ax, label="Count")
    plt.tight_layout()
    out_path = reports_dir / "temporal_model_predicted_vs_actual.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
