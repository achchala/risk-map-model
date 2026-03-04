#!/usr/bin/env python3
"""
Run diagnostics on the trained temporal crash model without starting the backend.

Uses the test-set results saved during training (same data the model was evaluated on):
- Target distribution, prediction vs actual, MAE/RMSE, baseline comparison,
  binned calibration, and top-K lift (mean actual in top 1%/5%/10% by pred vs overall).
  For sparse counts, ranking and lift matter more for routing than MAE alone.

Run from project root after training:
  python train_temporal_model.py   # saves temporal_model_test_results.npz
  python run_temporal_model_diagnostics.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Project root
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import OUTPUTS_DIR


def main() -> None:
    diagnostics_path = OUTPUTS_DIR / "reports" / "temporal_model_test_results.npz"

    if not diagnostics_path.exists():
        print(f"Test results not found: {diagnostics_path}")
        print("Run training first: python train_temporal_model.py")
        sys.exit(1)

    print("Loading test-set results from training run...")
    data = np.load(diagnostics_path)
    y_test = data["y_test"]
    y_pred = data["y_pred"]
    mean_train_y = float(data["mean_train_y"])
    y_pred = np.clip(y_pred, 0.0, None)
    n = len(y_test)
    print(f"Test set size: {n:,} (same sampled weekly panel used for model evaluation)")

    # ---- 1. Target distribution ----
    print("\n" + "=" * 60)
    print("1. TARGET (future_crash_count) on test set")
    print("=" * 60)
    print(f"  Count (n)     : {n:,}")
    print(f"  Mean          : {y_test.mean():.4f}")
    print(f"  Std           : {y_test.std():.4f}")
    print(f"  Min / Max     : {y_test.min():.0f} / {y_test.max():.0f}")
    pct_zero = (y_test == 0).mean() * 100
    pct_positive = (y_test > 0).mean() * 100
    print(f"  % zero        : {pct_zero:.1f}%")
    print(f"  % > 0         : {pct_positive:.1f}%")

    # ---- 2. Prediction distribution (including outlier impact) ----
    print("\n" + "=" * 60)
    print("2. PREDICTIONS (λ per segment-week) on test set")
    print("=" * 60)
    print(f"  Mean          : {y_pred.mean():.4f}")
    print(f"  Std           : {y_pred.std():.4f}")
    print(f"  Min / Max     : {y_pred.min():.4f} / {y_pred.max():.4f}")
    print(f"  Percentiles   : p50={np.percentile(y_pred, 50):.4f}, p90={np.percentile(y_pred, 90):.4f}, p99={np.percentile(y_pred, 99):.4f}, p99.9={np.percentile(y_pred, 99.9):.4f}")
    n_gt_1 = (y_pred > 1).sum()
    n_gt_5 = (y_pred > 5).sum()
    n_gt_10 = (y_pred > 10).sum()
    n_gt_100 = (y_pred > 100).sum()
    print(f"  Counts        : >1: {n_gt_1:,}, >5: {n_gt_5:,}, >10: {n_gt_10:,}, >100: {n_gt_100:,}")

    # ---- 3. Error metrics ----
    print("\n" + "=" * 60)
    print("3. ERROR METRICS (test set)")
    print("=" * 60)
    mae = np.mean(np.abs(y_pred - y_test))
    rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
    print(f"  MAE           : {mae:.4f}")
    print(f"  RMSE          : {rmse:.4f}")
    eps = 1e-9
    yt = np.maximum(y_test, eps)
    yp = np.maximum(y_pred, eps)
    poisson_dev = 2 * np.mean(yp - yt + yt * np.log(yt / yp))
    print(f"  Poisson dev   : {poisson_dev:.4f}")

    # ---- 4. Baseline: always predict mean(y_train) ----
    baseline_pred = np.full_like(y_test, mean_train_y)
    mae_baseline = np.mean(np.abs(baseline_pred - y_test))
    rmse_baseline = np.sqrt(np.mean((baseline_pred - y_test) ** 2))
    print("\n" + "=" * 60)
    print("4. BASELINE (predict mean train target)")
    print("=" * 60)
    print(f"  Mean(train y) : {mean_train_y:.4f}")
    print(f"  MAE (baseline): {mae_baseline:.4f}")
    print(f"  RMSE(baseline): {rmse_baseline:.4f}")
    print(f"  Model vs base : MAE {mae:.4f} vs {mae_baseline:.4f}  (lower is better)")

    # ---- 5. Correlation ----
    if n > 1 and y_test.std() > 0 and y_pred.std() > 0:
        corr = np.corrcoef(y_test, y_pred)[0, 1]
        print("\n" + "=" * 60)
        print("5. CORRELATION (actual vs predicted)")
        print("=" * 60)
        print(f"  Pearson r    : {corr:.4f}")

    # ---- 6. Binned calibration: bin by predicted λ, mean actual in each bin ----
    print("\n" + "=" * 60)
    print("6. BINNED CALIBRATION (predicted λ vs mean actual count)")
    print("=" * 60)
    n_bins = 5
    bins = np.percentile(y_pred, np.linspace(0, 100, n_bins + 1))
    bins[-1] = bins[-1] + 1e-9
    bin_idx = np.digitize(y_pred, bins) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() == 0:
            continue
        mean_pred = y_pred[mask].mean()
        mean_actual = y_test[mask].mean()
        count = mask.sum()
        print(f"  Bin {b+1}: n={count:,}, mean_pred={mean_pred:.4f}, mean_actual={mean_actual:.4f}")

    # ---- 7. Top-K lift: among top predicted rows, how much higher is actual crash rate? ----
    print("\n" + "=" * 60)
    print("7. TOP-K LIFT (model finding higher-risk windows?)")
    print("=" * 60)
    overall_mean = y_test.mean()
    print(f"  Overall mean(actual) : {overall_mean:.4f}")
    order = np.argsort(y_pred)[::-1]  # descending by prediction
    for pct in (1, 5, 10):
        k = max(1, int(n * pct / 100))
        top_k_idx = order[:k]
        mean_top_k = y_test[top_k_idx].mean()
        lift = mean_top_k / overall_mean if overall_mean > 0 else 0
        print(f"  Top {pct}% (n={k:,}): mean(actual)={mean_top_k:.4f}, lift={lift:.2f}x")

    print("\nDone. For routing, prioritize: top-K lift, calibration, correlation. MAE alone is misleading with 93%+ zeros.")


if __name__ == "__main__":
    main()
