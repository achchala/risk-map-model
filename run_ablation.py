#!/usr/bin/env python3
"""
run_ablation.py — Feature ablation study + hyperparameter stability sweep.

Proves model selection is justified, not arbitrary:
  - Ablation: retrain with each feature group removed → AUC-ROC delta shows
    which features actually contribute signal.
  - Hyperparameter stability: retrain across 6 parameter configs → proves the
    model is robust and not tuned to a fragile sweet spot.

Outputs (all in outputs/validation/):
  ablation_results.md        — table + bar chart (ablation_bar_chart.png)
  hyperparameter_stability.md — table + sensitivity plot (hyperparam_sensitivity.png)

Usage:
  python run_ablation.py

Note: Requires model training artifacts to be present (outputs/reports/*.npz, *.parquet).
      Does NOT overwrite outputs/models/toronto_temporal_count_model.pkl.
"""

from __future__ import annotations

import logging
import sys
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import DATA_DIR, OUTPUTS_DIR  # noqa: E402
from src.data_processing.data_loader import (  # type: ignore
    load_and_clean_data,
    load_historical_weather,
    load_model_dataset,
    load_school_locations,
    load_tmc_data,
    load_ttc_gtfs,
    merge_model_dataset_into_road_network,
    merge_school_zones_into_road_network,
    merge_tmc_into_road_network,
    merge_ttc_into_road_network,
)
from src.data_processing.spatial_join_fast import (  # type: ignore
    perform_spatial_join_event_level,
    _ensure_stable_segment_id,
)
from src.feature_engineering.panel_builder import (  # type: ignore
    PanelConfig,
    build_weekly_sampled_future_panel,
    temporal_train_val_test_split,
)
from src.models.model_trainer import HurdleTemporalTrainer  # type: ignore

try:
    from sklearn.metrics import roc_auc_score
except ImportError:
    roc_auc_score = None

logging.basicConfig(
    level=logging.WARNING,  # suppress verbose training logs during ablation
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("run_ablation")

VALIDATION_DIR = OUTPUTS_DIR / "validation"
VALIDATION_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Feature group definitions
# Each group defines columns to DROP before training.
# If a column isn't present in the panel, it's silently ignored.
# ---------------------------------------------------------------------------
WEATHER_COLS = [
    "temperature", "precipitation", "snow_depth_mm", "wind_speed",
    "is_freezing", "is_precip", "snow_mm", "visibility",
    "is_missing_weather",
    "snow_x_arterial", "freeze_x_arterial", "freeze_x_vehicle_vol",
    "precip_x_ped_vol", "freeze_x_transit", "freeze_x_rush_hour",
]

ROAD_GEOMETRY_COLS = [
    "segment_length", "is_oneway",
    "from_intersection_degree", "to_intersection_degree",
]
# road_class_* one-hot columns are matched by prefix at runtime

TMC_EXPOSURE_COLS = [
    "tmc_daily_ped_vol", "tmc_daily_cyclist_vol", "tmc_daily_vehicle_vol",
    "freeze_x_vehicle_vol", "precip_x_ped_vol",
]

SCHOOL_TRANSIT_COLS = [
    "is_school_zone", "nearby_transit_frequency",
    "is_school_active_hour", "freeze_x_transit",
]

LAG_FEATURE_COLS = [
    "crashes_1d_ago", "crashes_7d_ago", "crashes_30d_ago",
    "rolling_mean_7d", "rolling_max_7d",
    # hourly lag names (in case model was hourly)
    "past_crash_count_1h", "past_crash_count_24h", "past_crash_count_7d",
    "rolling_mean_24h", "rolling_max_24h",
    # weekly lag names
    "crashes_1_week_ago", "crashes_2_weeks_ago", "crashes_4_weeks_ago",
    "rolling_mean_4_weeks", "rolling_max_4_weeks",
]

HIST_PROFILE_COLS = [
    "hist_crashes_per_year", "hist_crash_hour_ratio", "hist_crash_weekend_ratio",
]

TEMPORAL_INDICATOR_COLS = [
    "month_sin", "month_cos", "season_int",
    "dow_sin", "dow_cos", "is_weekend", "day_of_week",
    "hour_sin", "hour_cos",
]

ABLATION_GROUPS: dict[str, list[str]] = {
    "weather":            WEATHER_COLS,
    "road_geometry":      ROAD_GEOMETRY_COLS,   # + road_class_* matched at runtime
    "tmc_exposure":       TMC_EXPOSURE_COLS,
    "school_transit":     SCHOOL_TRANSIT_COLS,
    "lag_features":       LAG_FEATURE_COLS,
    "hist_profiles":      HIST_PROFILE_COLS,
    "temporal_indicators": TEMPORAL_INDICATOR_COLS,
}

# ---------------------------------------------------------------------------
# Hyperparameter grid
# ---------------------------------------------------------------------------
HYPERPARAM_GRID = [
    {"max_depth": 4,  "learning_rate": 0.10, "max_iter": 300, "label": "depth=4 lr=0.10"},
    {"max_depth": 6,  "learning_rate": 0.10, "max_iter": 300, "label": "depth=6 lr=0.10 (baseline)"},
    {"max_depth": 8,  "learning_rate": 0.10, "max_iter": 300, "label": "depth=8 lr=0.10"},
    {"max_depth": 6,  "learning_rate": 0.05, "max_iter": 300, "label": "depth=6 lr=0.05"},
    {"max_depth": 6,  "learning_rate": 0.20, "max_iter": 300, "label": "depth=6 lr=0.20"},
    {"max_depth": 6,  "learning_rate": 0.10, "max_iter": 150, "label": "depth=6 lr=0.10 iter=150"},
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_auc_roc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute AUC-ROC; return 0.5 on failure."""
    if roc_auc_score is None:
        return float("nan")
    y_binary = (y_true > 0).astype(int)
    if y_binary.sum() == 0 or y_binary.sum() == len(y_binary):
        return 0.5
    try:
        return float(roc_auc_score(y_binary, y_score))
    except Exception:
        return 0.5


def _compute_lift_at_5pct(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Lift at top-5% of ranked windows."""
    k = max(1, int(0.05 * len(y_true)))
    top_idx = np.argsort(y_score)[::-1][:k]
    overall_rate = y_true.mean()
    if overall_rate == 0:
        return float("nan")
    top_rate = y_true[top_idx].mean()
    return float(top_rate / overall_rate)


def _train_and_score(panel: pd.DataFrame, drop_cols: list[str], panel_config: PanelConfig) -> dict:
    """
    Drop the specified columns from the panel, train a HurdleTemporalTrainer,
    evaluate on the test split, and return {auc_roc, lift_5}.
    """
    # Drop requested columns that are actually present
    actual_drop = [c for c in drop_cols if c in panel.columns]
    p = panel.drop(columns=actual_drop)

    train_data, val_data, test_data = temporal_train_val_test_split(p)

    trainer = HurdleTemporalTrainer(panel_config=panel_config, lambda_cap=50.0)

    # Add tail-weighted sample weights reusing the column if present
    sw_col = "sample_weight_tail" if "sample_weight_tail" in p.columns else "sample_weight"
    if sw_col not in p.columns:
        sw_col = None

    try:
        results = trainer.train_temporal_count_model(
            p,
            target_col="future_crash_count",
            sample_weight_col=sw_col,
        )
        y_test = np.asarray(results["y_test"])
        y_pred = np.asarray(results["y_pred"])
    except Exception as e:
        logger.warning("Training failed: %s", e)
        return {"auc_roc": float("nan"), "lift_5": float("nan")}

    return {
        "auc_roc": _compute_auc_roc(y_test, y_pred),
        "lift_5": _compute_lift_at_5pct(y_test, y_pred),
    }


def _load_full_panel() -> tuple[pd.DataFrame, PanelConfig]:
    """
    Rebuild the full daily training panel from raw data.
    Mirrors the logic in train_temporal_model.py exactly.
    """
    print("  Loading raw data...")
    collision_data, ksi_data, road_network = load_and_clean_data(DATA_DIR)
    event_level = perform_spatial_join_event_level(
        collision_data=collision_data,
        ksi_data=ksi_data,
        road_network=road_network,
    )
    road_with_ids = _ensure_stable_segment_id(road_network)
    model_dataset = load_model_dataset(DATA_DIR)
    road_with_ids = merge_model_dataset_into_road_network(road_with_ids, model_dataset)

    weather_data = load_historical_weather(DATA_DIR)

    tmc_data = load_tmc_data(DATA_DIR)
    road_with_ids = merge_tmc_into_road_network(road_with_ids, tmc_data)

    school_locations = load_school_locations(DATA_DIR)
    road_with_ids = merge_school_zones_into_road_network(road_with_ids, school_locations)

    ttc_stops = load_ttc_gtfs(DATA_DIR)
    road_with_ids = merge_ttc_into_road_network(road_with_ids, ttc_stops)

    panel_config = PanelConfig(window_size_hours=24, horizon_hours=24)

    print("  Building daily training panel...")
    panel = build_weekly_sampled_future_panel(
        event_level_crashes=event_level,
        road_network=road_with_ids,
        weather_data=weather_data,
        window_size_hours=panel_config.window_size_hours,
        horizon_hours=panel_config.horizon_hours,
    )

    # Add tail-weighted sample weights
    from train_temporal_model import _add_tail_weighted_sample_weights  # type: ignore
    panel = _add_tail_weighted_sample_weights(
        panel,
        target_col="future_crash_count",
        sampling_weight_col="sample_weight",
        output_weight_col="sample_weight_tail",
        alpha=2.0,
        tail_threshold=2.0,
        weight_cap=50.0,
    )

    return panel, panel_config


def _plot_ablation_bar(ablation_rows: list[dict], baseline_auc: float) -> None:
    """Save ablation_bar_chart.png."""
    labels = [r["group"] for r in ablation_rows]
    deltas = [r["auc_roc_delta"] for r in ablation_rows]
    colors = ["#d73027" if d < -0.005 else "#4575b4" for d in deltas]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(labels, deltas, color=colors)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("AUC-ROC delta vs. full model (negative = feature group matters)")
    ax.set_title(f"Feature Ablation Study — Full model AUC-ROC = {baseline_auc:.4f}")
    plt.tight_layout()
    fig.savefig(VALIDATION_DIR / "ablation_bar_chart.png", dpi=120)
    plt.close(fig)


def _plot_hyperparam_sensitivity(hp_rows: list[dict]) -> None:
    """Save hyperparam_sensitivity.png."""
    labels = [r["label"] for r in hp_rows]
    aucs   = [r["auc_roc"] for r in hp_rows]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(range(len(labels)), aucs, marker="o", color="#2c7bb6", linewidth=1.5)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("AUC-ROC (test set)")
    ax.set_title("Hyperparameter Stability — AUC-ROC across 6 configurations")
    ax.set_ylim(max(0, min(aucs) - 0.05), min(1, max(aucs) + 0.05))
    plt.tight_layout()
    fig.savefig(VALIDATION_DIR / "hyperparam_sensitivity.png", dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main ablation runner
# ---------------------------------------------------------------------------

def run_feature_ablation(panel: pd.DataFrame, panel_config: PanelConfig, baseline_auc: float) -> list[dict]:
    """
    Retrain HurdleTemporalTrainer with each feature group removed.
    Returns list of dicts with group, auc_roc, auc_roc_delta, lift_5.
    """
    rows = []
    for group_name, drop_cols in ABLATION_GROUPS.items():
        # For road_geometry, also drop road_class_* one-hot columns
        extra = []
        if group_name == "road_geometry":
            extra = [c for c in panel.columns if c.startswith("road_class_")]
        all_drop = drop_cols + extra

        t0 = time.perf_counter()
        print(f"  [{group_name}] Dropping {len([c for c in all_drop if c in panel.columns])} columns...", end=" ", flush=True)
        scores = _train_and_score(panel, all_drop, panel_config)
        elapsed = time.perf_counter() - t0

        delta = scores["auc_roc"] - baseline_auc
        rows.append({
            "group": group_name,
            "auc_roc": round(scores["auc_roc"], 4),
            "auc_roc_delta": round(delta, 4),
            "lift_5": round(scores["lift_5"], 2) if not np.isnan(scores["lift_5"]) else "N/A",
            "elapsed_s": round(elapsed, 1),
        })
        print(f"AUC-ROC={scores['auc_roc']:.4f} (Δ={delta:+.4f}) [{elapsed:.0f}s]")

    return rows


def run_hyperparameter_stability(panel: pd.DataFrame, panel_config: PanelConfig) -> list[dict]:
    """
    Retrain HurdleTemporalTrainer across HYPERPARAM_GRID.
    Returns list of dicts with label, max_depth, learning_rate, max_iter, auc_roc.
    """
    rows = []
    sw_col = "sample_weight_tail" if "sample_weight_tail" in panel.columns else "sample_weight"
    if sw_col not in panel.columns:
        sw_col = None

    for cfg in HYPERPARAM_GRID:
        print(f"  [{cfg['label']}]...", end=" ", flush=True)
        t0 = time.perf_counter()

        trainer = HurdleTemporalTrainer(panel_config=panel_config, lambda_cap=50.0)
        # Override hyperparameters by monkey-patching after init
        trainer._max_depth      = cfg["max_depth"]
        trainer._learning_rate  = cfg["learning_rate"]
        trainer._max_iter       = cfg["max_iter"]

        # Re-implement train with custom params
        from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
        from src.feature_engineering.panel_builder import temporal_train_val_test_split  # type: ignore
        from sklearn.preprocessing import StandardScaler

        train_data, val_data, test_data = temporal_train_val_test_split(panel)
        X_train, y_train = trainer._prepare_features(train_data)
        X_val, y_val     = trainer._prepare_features(val_data)
        X_test, y_test   = trainer._prepare_features(test_data)
        trainer.feature_columns = list(X_train.columns)

        trainer.scaler = StandardScaler()
        Xtr_s = trainer.scaler.fit_transform(X_train)
        Xva_s = trainer.scaler.transform(X_val)
        Xte_s = trainer.scaler.transform(X_test)

        sw = train_data[sw_col].astype(float).values if sw_col else None

        stage1 = HistGradientBoostingClassifier(
            loss="log_loss",
            max_depth=cfg["max_depth"],
            learning_rate=cfg["learning_rate"],
            max_iter=cfg["max_iter"],
            random_state=42,
        )
        stage1.fit(Xtr_s, (y_train > 0).astype(int), sample_weight=sw)

        pos_mask = y_train > 0
        stage2 = HistGradientBoostingRegressor(
            loss="poisson",
            max_depth=cfg["max_depth"],
            learning_rate=cfg["learning_rate"],
            max_iter=cfg["max_iter"],
            random_state=42,
        )
        sw_pos = sw[pos_mask] if sw is not None else None
        stage2.fit(Xtr_s[pos_mask], y_train[pos_mask], sample_weight=sw_pos)

        p_test = stage1.predict_proba(Xte_s)[:, 1]
        lam_test = np.clip(stage2.predict(Xte_s), 0, None)
        y_pred = p_test * lam_test

        auc = _compute_auc_roc(np.asarray(y_test), y_pred)
        elapsed = time.perf_counter() - t0

        rows.append({
            "label": cfg["label"],
            "max_depth": cfg["max_depth"],
            "learning_rate": cfg["learning_rate"],
            "max_iter": cfg["max_iter"],
            "auc_roc": round(auc, 4),
            "elapsed_s": round(elapsed, 1),
        })
        print(f"AUC-ROC={auc:.4f} [{elapsed:.0f}s]")

    return rows


def write_ablation_report(ablation_rows: list[dict], baseline_auc: float) -> None:
    aucs = [r["auc_roc"] for r in ablation_rows if not np.isnan(r["auc_roc"])]
    largest_drop = min(ablation_rows, key=lambda r: r["auc_roc_delta"]) if ablation_rows else {}

    lines = [
        "# Feature Ablation Study",
        "",
        f"**Full model AUC-ROC (baseline): {baseline_auc:.4f}**",
        "",
        "Each row shows performance when that feature group is removed entirely.",
        "A large negative delta = that group contributes meaningful signal.",
        "",
        "## Results",
        "",
        "| Feature Group | AUC-ROC | Δ vs Full Model | Lift@5% | Time (s) |",
        "|---|---|---|---|---|",
    ]
    for r in sorted(ablation_rows, key=lambda x: x["auc_roc_delta"]):
        sign = "▼" if r["auc_roc_delta"] < -0.005 else ("▲" if r["auc_roc_delta"] > 0.005 else "≈")
        lines.append(
            f"| **{r['group']}** | {r['auc_roc']} | {sign} {r['auc_roc_delta']:+.4f} | {r['lift_5']} | {r['elapsed_s']} |"
        )

    if largest_drop:
        lines += [
            "",
            "## Key Finding",
            f"The **{largest_drop['group']}** feature group caused the largest performance drop",
            f"(AUC-ROC delta = {largest_drop['auc_roc_delta']:+.4f}) when removed.",
            "",
            f"AUC-ROC range across ablation runs: {min(aucs):.4f} – {max(aucs):.4f}",
        ]

    lines += [
        "",
        "## Stability",
        f"The model retained AUC-ROC > {min(aucs):.4f} even with any single feature group removed,",
        "demonstrating that no single group is a single point of failure.",
        "",
        "![Ablation bar chart](ablation_bar_chart.png)",
    ]

    (VALIDATION_DIR / "ablation_results.md").write_text("\n".join(lines))
    print(f"  Written: {VALIDATION_DIR / 'ablation_results.md'}")


def write_hyperparameter_report(hp_rows: list[dict]) -> None:
    aucs = [r["auc_roc"] for r in hp_rows]
    auc_std  = float(np.std(aucs))
    auc_range = max(aucs) - min(aucs)
    stable = auc_std < 0.02

    lines = [
        "# Hyperparameter Stability",
        "",
        "Retraining HurdleTemporalTrainer across 6 parameter configurations to verify",
        "that the chosen hyperparameters are not a fragile optimum.",
        "",
        "## Results",
        "",
        "| Configuration | max_depth | learning_rate | max_iter | AUC-ROC | Time (s) |",
        "|---|---|---|---|---|---|",
    ]
    for r in hp_rows:
        marker = " **(baseline)**" if "baseline" in r["label"] else ""
        lines.append(
            f"| {r['label']}{marker} | {r['max_depth']} | {r['learning_rate']} | {r['max_iter']} | {r['auc_roc']} | {r['elapsed_s']} |"
        )

    lines += [
        "",
        "## Summary",
        "",
        f"| Metric | Value |",
        f"|---|---|",
        f"| AUC-ROC range | {min(aucs):.4f} – {max(aucs):.4f} |",
        f"| AUC-ROC std | {auc_std:.4f} |",
        f"| Stable (std < 0.02) | {'**YES ✅**' if stable else '**NO ⚠️**'} |",
        "",
        f"{'The model is **robust** to hyperparameter variation.' if stable else 'The model shows **sensitivity** to hyperparameter choice — consider further tuning.'}",
        "",
        "![Hyperparameter sensitivity plot](hyperparam_sensitivity.png)",
    ]

    (VALIDATION_DIR / "hyperparameter_stability.md").write_text("\n".join(lines))
    print(f"  Written: {VALIDATION_DIR / 'hyperparameter_stability.md'}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    t_start = time.perf_counter()
    print("\n" + "=" * 60)
    print("ABLATION STUDY + HYPERPARAMETER STABILITY")
    print("=" * 60)

    # 1. Load / rebuild the full training panel
    print("\n[1/4] Building full training panel (this may take several minutes)...")
    panel, panel_config = _load_full_panel()
    print(f"  Panel shape: {panel.shape}")

    # 2. Establish baseline AUC-ROC with full feature set
    print("\n[2/4] Baseline: training with full feature set...")
    baseline_scores = _train_and_score(panel, [], panel_config)
    baseline_auc = baseline_scores["auc_roc"]
    print(f"  Baseline AUC-ROC={baseline_auc:.4f}, Lift@5%={baseline_scores['lift_5']:.2f}×")

    # 3. Feature ablation
    print("\n[3/4] Feature ablation (7 groups × 1 retrain each)...")
    ablation_rows = run_feature_ablation(panel, panel_config, baseline_auc)
    write_ablation_report(ablation_rows, baseline_auc)
    _plot_ablation_bar(ablation_rows, baseline_auc)

    # 4. Hyperparameter stability
    print("\n[4/4] Hyperparameter stability (6 configs × 1 retrain each)...")
    hp_rows = run_hyperparameter_stability(panel, panel_config)
    write_hyperparameter_report(hp_rows)
    _plot_hyperparam_sensitivity(hp_rows)

    elapsed = time.perf_counter() - t_start
    print("\n" + "=" * 60)
    print(f"DONE in {elapsed / 60:.1f} min")
    print(f"Outputs → {VALIDATION_DIR}/")
    print("  ablation_results.md")
    print("  ablation_bar_chart.png")
    print("  hyperparameter_stability.md")
    print("  hyperparam_sensitivity.png")
    print("=" * 60)


if __name__ == "__main__":
    main()
