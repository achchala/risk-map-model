#!/usr/bin/env python3
"""
Comprehensive evaluation of the trained temporal crash prediction model.

Loads test-set results from training and computes:
- Ranking effectiveness (AUC-PR, AUC-ROC, lift, recall at K)
- Calibration (Brier score, calibration error, binned reliability)
- Regression metrics (MAE, RMSE, Poisson deviance) with baseline comparison
- Binary view (treating "any crash" as positive) for interpretability
- Written report with verdict and modification recommendations

Run from project root after training:
  python train_temporal_model.py
  python evaluate_temporal_model.py

Outputs:
  - Console summary
  - outputs/reports/temporal_model_evaluation_report.md
  - Optional: outputs/reports/temporal_model_evaluation_plots.png (if matplotlib available)
"""

from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime

import numpy as np

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import OUTPUTS_DIR


def _safe_auc_roc(y_true_binary: np.ndarray, y_score: np.ndarray) -> float:
    """AUC-ROC; returns 0.5 if undefined."""
    try:
        from sklearn.metrics import roc_auc_score
        if y_true_binary.sum() == 0 or y_true_binary.sum() == len(y_true_binary):
            return 0.5
        return float(roc_auc_score(y_true_binary, y_score))
    except Exception:
        return 0.5


def _safe_auc_pr(y_true_binary: np.ndarray, y_score: np.ndarray) -> float:
    """Average precision (AUC-PR); returns 0 if undefined."""
    try:
        from sklearn.metrics import average_precision_score
        if y_true_binary.sum() == 0:
            return 0.0
        return float(average_precision_score(y_true_binary, y_score))
    except Exception:
        return 0.0


def _brier_score(y_true_binary: np.ndarray, p_prob: np.ndarray) -> float:
    """Brier score for P(>=1 crash); lower is better."""
    p_prob = np.clip(p_prob, 1e-6, 1 - 1e-6)
    return float(np.mean((p_prob - y_true_binary) ** 2))


def _ece_binned(y_true_binary: np.ndarray, p_prob: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error (ECE): mean absolute difference between mean predicted prob and mean actual in each bin."""
    bins = np.linspace(0, 1, n_bins + 1)
    bins[-1] = 1.01
    bin_idx = np.digitize(np.clip(p_prob, 0, 1), bins) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    ece = 0.0
    total = 0
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() == 0:
            continue
        mean_pred = p_prob[mask].mean()
        mean_actual = y_true_binary[mask].mean()
        ece += mask.sum() * abs(mean_pred - mean_actual)
        total += mask.sum()
    return ece / total if total > 0 else 0.0


def run_evaluation() -> dict:
    """Load test results, compute all metrics, return a dict for reporting."""
    reports_dir = OUTPUTS_DIR / "reports"
    diagnostics_path = reports_dir / "temporal_model_test_results.npz"

    if not diagnostics_path.exists():
        raise FileNotFoundError(
            f"Test results not found: {diagnostics_path}\nRun: python train_temporal_model.py"
        )

    data = np.load(diagnostics_path)
    y_test = np.asarray(data["y_test"], dtype=float)
    y_pred = np.asarray(data["y_pred"], dtype=float)
    y_pred = np.clip(y_pred, 0.0, None)
    sample_weight_test = (
        data["sample_weight_test"]
        if "sample_weight_test" in data.files
        else np.ones_like(y_test, dtype=float)
    )
    mean_train_y = float(data["mean_train_y"])

    n = len(y_test)
    binary = (y_test > 0).astype(int)  # 1 if any crash, 0 otherwise
    n_pos = int(binary.sum())
    p_prob = 1.0 - np.exp(-y_pred)  # P(at least one crash) under Poisson

    # ---- Regression ----
    mae = float(np.mean(np.abs(y_pred - y_test)))
    rmse = float(np.sqrt(np.mean((y_pred - y_test) ** 2)))
    eps = 1e-9
    yt = np.maximum(y_test, eps)
    yp = np.maximum(y_pred, eps)
    poisson_dev = float(2 * np.mean(yp - yt + yt * np.log(yt / yp)))
    baseline_pred = np.full_like(y_test, mean_train_y)
    mae_baseline = float(np.mean(np.abs(baseline_pred - y_test)))
    rmse_baseline = float(np.sqrt(np.mean((baseline_pred - y_test) ** 2)))

    # ---- Ranking (binary: any crash vs none) ----
    auc_roc = _safe_auc_roc(binary, y_pred)
    auc_pr = _safe_auc_pr(binary, y_pred)
    brier = _brier_score(binary, p_prob)
    ece = _ece_binned(binary, p_prob, n_bins=10)

    # Baseline AUC-PR: random classifier has AUC-PR = prevalence
    prevalence = binary.mean()
    auc_pr_baseline = prevalence

    # ---- Lift at K ----
    order = np.argsort(y_pred)[::-1]
    overall_mean = y_test.mean()
    lift_at_k = {}
    recall_at_k = {}
    for pct in (1, 2, 5, 10, 20):
        k = max(1, int(n * pct / 100))
        top_k_idx = order[:k]
        mean_top = y_test[top_k_idx].mean()
        lift_at_k[pct] = (mean_top / overall_mean) if overall_mean > 0 else 0.0
        captured = binary[top_k_idx].sum()
        recall_at_k[pct] = (captured / n_pos) if n_pos > 0 else 0.0

    # ---- Tail capture (y>=1, y>=2) ----
    for thresh in (1, 2):
        pos = (y_test >= thresh)
        n_p = int(pos.sum())
        if n_p == 0:
            continue
        for pct in (1, 5, 10):
            k = max(1, int(n * pct / 100))
            cap = pos[order[:k]].sum()
            recall_at_k.setdefault(f"y>={thresh}_top{pct}pct", cap / n_p if n_p > 0 else 0)

    # ---- Calibration bins (by predicted lambda) ----
    n_bins_cal = 10
    bins = np.percentile(y_pred, np.linspace(0, 100, n_bins_cal + 1))
    bins[-1] = bins[-1] + 1e-9
    bin_idx = np.digitize(y_pred, bins) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins_cal - 1)
    calibration_bins = []
    for b in range(n_bins_cal):
        mask = bin_idx == b
        if mask.sum() == 0:
            continue
        calibration_bins.append({
            "bin": b + 1,
            "n": int(mask.sum()),
            "mean_pred": float(y_pred[mask].mean()),
            "mean_actual": float(y_test[mask].mean()),
            "mean_actual_weighted": float(np.average(y_test[mask], weights=sample_weight_test[mask])),
        })

    return {
        "n": n,
        "n_positive": n_pos,
        "prevalence": prevalence,
        "y_test_mean": float(y_test.mean()),
        "y_test_pct_zero": float((y_test == 0).mean() * 100),
        "y_pred_mean": float(y_pred.mean()),
        "y_pred_median": float(np.median(y_pred)),
        "y_pred_p99": float(np.percentile(y_pred, 99)),
        "mae": mae,
        "rmse": rmse,
        "poisson_deviance": poisson_dev,
        "mae_baseline": mae_baseline,
        "rmse_baseline": rmse_baseline,
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
        "auc_pr_baseline": auc_pr_baseline,
        "brier": brier,
        "ece": ece,
        "lift_at_k": lift_at_k,
        "recall_at_k": recall_at_k,
        "calibration_bins": calibration_bins,
        "mean_train_y": mean_train_y,
    }


def write_report(metrics: dict, out_path: Path) -> None:
    """Write markdown evaluation report with verdict and recommendations."""
    m = metrics
    lines = [
        "# Temporal Crash Model — Evaluation Report",
        "",
        f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*",
        "",
        "---",
        "",
        "## 1. Test Set Summary",
        "",
        f"- **Test size:** {m['n']:,} rows (segment–hour windows)",
        f"- **Positive rate (any crash):** {m['n_positive']:,} ({m['prevalence']*100:.2f}%)",
        f"- **Target mean:** {m['y_test_mean']:.4f} (crashes per segment-hour)",
        f"- **Target % zero:** {m['y_test_pct_zero']:.1f}%",
        "",
        "---",
        "",
        "## 2. Prediction Distribution",
        "",
        f"- **Mean prediction (λ):** {m['y_pred_mean']:.4f}",
        f"- **Median prediction:** {m['y_pred_median']:.4f}",
        f"- **99th percentile:** {m['y_pred_p99']:.4f}",
        "",
        "---",
        "",
        "## 3. Regression Metrics",
        "",
        "| Metric | Model | Baseline (predict mean) | Note |",
        "|--------|--------|---------------------------|------|",
        f"| MAE | {m['mae']:.4f} | {m['mae_baseline']:.4f} | Lower is better; baseline = constant mean |",
        f"| RMSE | {m['rmse']:.4f} | {m['rmse_baseline']:.4f} | Penalizes large errors |",
        f"| Poisson deviance | {m['poisson_deviance']:.4f} | — | Fit of count model |",
        "",
        "**Interpretation:** With 99%+ zeros, MAE/RMSE are dominated by prediction magnitude. A model that predicts small positive λ for many rows can have higher MAE than always predicting the (near-zero) mean. Prefer ranking and calibration for sparse counts.",
        "",
        "---",
        "",
        "## 4. Ranking Effectiveness (Binary: Any Crash vs None)",
        "",
        "| Metric | Value | Baseline / Note |",
        "|--------|--------|------------------|",
        f"| **AUC-ROC** | {m['auc_roc']:.4f} | 0.5 = random; >0.5 = model ranks positives higher |",
        f"| **AUC-PR (Average Precision)** | {m['auc_pr']:.4f} | Baseline ~ prevalence ({m['auc_pr_baseline']:.4f}); higher = better ranking |",
        "",
        "### Lift at Top K%",
        "",
        "Among the top K% of rows by predicted risk, how much higher is the actual crash rate than average?",
        "",
        "| Top K% | Lift (mean actual / overall mean) | Recall (of all positives in top K%) |",
        "|--------|-------------------------------------|--------------------------------------|",
    ]
    for pct in (1, 2, 5, 10, 20):
        lift = m["lift_at_k"].get(pct, 0)
        rec = m["recall_at_k"].get(pct, 0)
        lines.append(f"| Top {pct}% | {lift:.2f}x | {rec*100:.1f}% |")
    lines.extend([
        "",
        "**Interpretation:** Lift > 1 means the model concentrates actual crashes in the segments it flags as high-risk. This is what matters for routing: avoiding the top-ranked segments should reduce exposure.",
        "",
        "---",
        "",
        "## 5. Calibration",
        "",
        f"- **Brier score (P(≥1 crash)):** {m['brier']:.4f} (lower is better; 0.25 = no skill)",
        f"- **Expected Calibration Error (ECE):** {m['ece']:.4f} (lower = predicted probabilities closer to actual frequencies)",
        "",
        "### Binned Calibration (by predicted λ)",
        "",
        "| Bin | n | Mean predicted λ | Mean actual count |",
        "|-----|-----|-------------------|-------------------|",
    ])
    for cb in m["calibration_bins"][:10]:
        lines.append(f"| {cb['bin']} | {cb['n']:,} | {cb['mean_pred']:.4f} | {cb['mean_actual']:.4f} |")
    lines.extend([
        "",
        "---",
        "",
        "## 6. Verdict",
        "",
    ])

    # Verdict logic
    verdict_parts = []
    if m["auc_pr"] > m["auc_pr_baseline"] * 1.5:
        verdict_parts.append("- **Ranking:** Model ranks meaningfully better than random (AUC-PR > 1.5× prevalence).")
    elif m["auc_pr"] > m["auc_pr_baseline"]:
        verdict_parts.append("- **Ranking:** Model has some ranking signal (AUC-PR > prevalence) but could be improved.")
    else:
        verdict_parts.append("- **Ranking:** Model ranking is weak; consider feature or target changes.")

    lift_5 = m["lift_at_k"].get(5, 0)
    if lift_5 >= 5:
        verdict_parts.append(f"- **Lift:** Strong lift at top 5% ({lift_5:.1f}x) — useful for safety-aware routing.")
    elif lift_5 >= 2:
        verdict_parts.append(f"- **Lift:** Moderate lift at top 5% ({lift_5:.1f}x) — some value for routing.")
    else:
        verdict_parts.append(f"- **Lift:** Low lift at top 5% ({lift_5:.1f}x) — limited discrimination.")

    if m["ece"] < 0.05:
        verdict_parts.append(f"- **Calibration:** Good (ECE = {m['ece']:.3f}).")
    elif m["ece"] < 0.15:
        verdict_parts.append(f"- **Calibration:** Acceptable (ECE = {m['ece']:.3f}); consider isotonic recalibration.")
    else:
        verdict_parts.append(f"- **Calibration:** Poor (ECE = {m['ece']:.3f}); probabilities should not be trusted without recalibration.")

    if m["mae"] > m["mae_baseline"]:
        verdict_parts.append("- **MAE:** Model MAE is higher than baseline (predict mean). This is common with sparse counts when the model assigns non-zero risk to many rows; focus on ranking and lift instead.")

    lines.extend(verdict_parts)
    lines.extend([
        "",
        "---",
        "",
        "## 7. Recommendations",
        "",
        "| If... | Then consider... |",
        "|-------|-------------------|",
        "| AUC-PR or lift is low | Add or improve features (e.g. hourly weather, more traffic coverage); try daily instead of hourly windows to reduce sparsity. |",
        "| Calibration (ECE) is high | Apply or retrain isotonic calibration on validation set; report P(≥1 crash) from calibrated probabilities. |",
        "| MAE worse than baseline | Ignore MAE for decision-making; use lift and AUC-PR. Optionally try Negative Binomial or zero-inflated model if overdispersion is confirmed at coarser granularity. |",
        "| Top 1% recall is low | Increase model capacity (depth/trees) or tail weighting; check that high-crash segments have sufficient features (e.g. traffic volume). |",
        "| Model is good enough for routing | Deploy; monitor lift and calibration on a holdout period. |",
        "",
    ])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    reports_dir = OUTPUTS_DIR / "reports"
    diagnostics_path = reports_dir / "temporal_model_test_results.npz"

    if not diagnostics_path.exists():
        print(f"Test results not found: {diagnostics_path}")
        print("Run training first: python train_temporal_model.py")
        sys.exit(1)

    print("Loading test-set results and computing metrics...")
    metrics = run_evaluation()

    # Console summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  Test n           : {metrics['n']:,}")
    print(f"  Positives (y>0)   : {metrics['n_positive']:,} ({metrics['prevalence']*100:.2f}%)")
    print(f"  AUC-ROC           : {metrics['auc_roc']:.4f}  (0.5=random)")
    print(f"  AUC-PR            : {metrics['auc_pr']:.4f}  (baseline~{metrics['auc_pr_baseline']:.4f})")
    print(f"  Brier score       : {metrics['brier']:.4f}")
    print(f"  ECE (calibration) : {metrics['ece']:.4f}")
    print(f"  MAE  (model/base) : {metrics['mae']:.4f} / {metrics['mae_baseline']:.4f}")
    print(f"  Lift @ top 5%     : {metrics['lift_at_k'].get(5,0):.2f}x")
    print(f"  Recall @ top 5%   : {metrics['recall_at_k'].get(5,0)*100:.1f}%")
    print("=" * 60)

    report_path = reports_dir / "temporal_model_evaluation_report.md"
    write_report(metrics, report_path)
    print(f"\nReport written: {report_path}")

    # Optional: plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        y_test = np.load(diagnostics_path)["y_test"]
        y_pred = np.clip(np.load(diagnostics_path)["y_pred"], 0, None)
        binary = (y_test > 0).astype(int)

        # 1. Predicted vs actual (hexbin, capped)
        ax = axes[0, 0]
        xp = np.clip(y_pred, 0, 10)
        yp = np.clip(y_test, 0, 10)
        ax.hexbin(xp, yp, gridsize=30, mincnt=1, cmap="viridis", edgecolors="none")
        ax.plot([0, 10], [0, 10], "r--", lw=1.5, label="Perfect")
        ax.set_xlabel("Predicted λ")
        ax.set_ylabel("Actual count")
        ax.set_title("Predicted vs actual")
        ax.legend()

        # 2. Calibration curve (mean actual vs mean pred in bins)
        ax = axes[0, 1]
        p_prob = 1 - np.exp(-y_pred)
        bins = np.percentile(p_prob, np.linspace(0, 100, 11))
        bins[-1] += 1e-9
        bin_idx = np.digitize(p_prob, bins) - 1
        bin_idx = np.clip(bin_idx, 0, 9)
        bin_means_pred = []
        bin_means_actual = []
        for b in range(10):
            m = bin_idx == b
            if m.sum() > 0:
                bin_means_pred.append(p_prob[m].mean())
                bin_means_actual.append(binary[m].mean())
        if bin_means_pred:
            ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect")
            ax.plot(bin_means_pred, bin_means_actual, "o-", label="Model")
            ax.set_xlabel("Mean predicted P(≥1 crash)")
            ax.set_ylabel("Mean actual (binary)")
            ax.set_title("Calibration curve")
            ax.legend()
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)

        # 3. Lift curve (cumulative recall vs fraction of population)
        ax = axes[1, 0]
        n_plot = len(y_pred)
        order = np.argsort(y_pred)[::-1]
        frac = np.arange(1, n_plot + 1) / n_plot
        cum_captured = np.cumsum(binary[order])
        recall_curve = cum_captured / binary.sum() if binary.sum() > 0 else np.zeros_like(frac)
        ax.plot(frac, recall_curve, label="Model")
        ax.plot([0, 1], [0, 1], "k--", label="Random")
        ax.set_xlabel("Fraction of population (by risk rank)")
        ax.set_ylabel("Cumulative recall (fraction of positives)")
        ax.set_title("Lift curve (cumulative recall)")
        ax.legend()

        # 4. Top-K lift bar chart
        ax = axes[1, 1]
        pcts = list(metrics["lift_at_k"].keys())
        lifts = [metrics["lift_at_k"][p] for p in pcts]
        ax.bar([f"Top {p}%" for p in pcts], lifts, color="steelblue", edgecolor="black")
        ax.axhline(1, color="gray", linestyle="--")
        ax.set_ylabel("Lift (vs overall mean)")
        ax.set_title("Lift at top K%")
        plt.tight_layout()
        plot_path = reports_dir / "temporal_model_evaluation_plots.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Plots saved: {plot_path}")
    except Exception as e:
        print(f"Plots skipped: {e}")

    print("\nDone. Open the report for full verdict and recommendations.")


if __name__ == "__main__":
    main()
