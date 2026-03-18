#!/usr/bin/env python3
"""
Train the temporal crash likelihood model and save:
- the trained TemporalCountModelTrainer
- the latest panel snapshot for inference

Outputs:
- outputs/models/toronto_temporal_count_model.pkl
- outputs/reports/panel_latest.parquet
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Ensure project root is on path when run from any directory
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import pandas as pd

from config import (
    DATA_DIR,
    OUTPUTS_DIR,
)

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
    build_latest_window_inference_panel,
    build_weekly_sampled_future_panel,
    temporal_train_val_test_split,
)
from src.models.model_trainer import HurdleTemporalTrainer, TemporalCountModelTrainer  # type: ignore


def _add_tail_weighted_sample_weights(
    panel: pd.DataFrame,
    target_col: str = "future_crash_count",
    sampling_weight_col: str = "sample_weight",
    output_weight_col: str = "sample_weight_tail",
    alpha: float = 2.0,
    tail_threshold: float = 2.0,
    weight_cap: float = 50.0,
) -> pd.DataFrame:
    """
    Add tail-weighted sample weights as:
        w_final = w_sampling * w_tail(y)

    Tail emphasis is applied using a bounded log-shape for y >= tail_threshold,
    then normalized on the train split so mean(w_final_train) matches
    mean(w_sampling_train), and finally capped for numerical stability.
    """
    panel = panel.copy()
    if sampling_weight_col not in panel.columns:
        raise ValueError(
            f"Sampling weight column '{sampling_weight_col}' not found in panel."
        )
    if target_col not in panel.columns:
        raise ValueError(f"Target column '{target_col}' not found in panel.")

    # Use the same temporal split used in training so diagnostics are train-only.
    train_data, _, _ = temporal_train_val_test_split(panel)
    train_windows = train_data["window_start"].unique()
    train_mask = panel["window_start"].isin(train_windows)

    y = panel[target_col].astype(float).values
    w_sampling = panel[sampling_weight_col].astype(float).values

    w_tail = np.ones_like(y, dtype=float)
    tail_mask = y >= tail_threshold
    w_tail[tail_mask] = 1.0 + alpha * np.log1p(y[tail_mask])

    w_final = w_sampling * w_tail

    # Normalize by train split only so overall effective scale stays stable.
    mean_sampling_train = float(np.mean(w_sampling[train_mask]))
    mean_final_train = float(np.mean(w_final[train_mask]))
    if mean_final_train > 0:
        w_final *= mean_sampling_train / mean_final_train

    w_final = np.clip(w_final, 0.0, weight_cap)
    panel[output_weight_col] = w_final

    # Log train-label distribution for QA and safe threshold tuning.
    y_train = panel.loc[train_mask, target_col].astype(float)
    bucket_counts = {
        "y=0": int((y_train == 0).sum()),
        "y=1": int((y_train == 1).sum()),
        "y=2": int((y_train == 2).sum()),
        "y=3": int((y_train == 3).sum()),
        "y=4": int((y_train == 4).sum()),
        "y>=5": int((y_train >= 5).sum()),
    }
    logger = logging.getLogger("train_temporal_model")
    logger.info("Train future_crash_count buckets: %s", bucket_counts)
    logger.info(
        "Tail weighting config: threshold=%.1f, alpha=%.2f, cap=%.1f",
        tail_threshold,
        alpha,
        weight_cap,
    )
    logger.info(
        "Weight means (train): sampling=%.4f, tail_weighted=%.4f",
        mean_sampling_train,
        float(np.mean(panel.loc[train_mask, output_weight_col])),
    )
    logger.info(
        "Tail-weighted train percentiles: p50=%.4f, p90=%.4f, p99=%.4f, max=%.4f",
        float(np.percentile(panel.loc[train_mask, output_weight_col], 50)),
        float(np.percentile(panel.loc[train_mask, output_weight_col], 90)),
        float(np.percentile(panel.loc[train_mask, output_weight_col], 99)),
        float(np.max(panel.loc[train_mask, output_weight_col])),
    )

    return panel


def _log_panel_feature_summary(panel: pd.DataFrame, logger: logging.Logger) -> None:
    """Log a concise feature summary so we can verify the model sees real variation."""
    logger.info("--- Panel Feature Summary ---")
    logger.info("Columns (%d): %s", len(panel.columns), list(panel.columns))

    if "window_start" in panel.columns:
        ws = pd.to_datetime(panel["window_start"])
        logger.info(
            "window_start range: %s to %s, %d unique values",
            ws.min(),
            ws.max(),
            ws.nunique(),
        )
        if "hour_of_day" in panel.columns:
            logger.info(
                "hour_of_day unique values: %s", sorted(panel["hour_of_day"].unique())
            )
        if "day_of_week" in panel.columns:
            logger.info(
                "day_of_week unique values: %s", sorted(panel["day_of_week"].unique())
            )

    rc_cols = [c for c in panel.columns if c.startswith("road_class_")]
    logger.info("road_class one-hot columns (%d): %s", len(rc_cols), rc_cols)

    for col in [
        "is_oneway",
        "from_intersection_degree",
        "to_intersection_degree",
        "segment_length",
        "hist_crashes_per_year",
        "hist_crash_hour_ratio",
        "hist_crash_weekend_ratio",
        "is_missing_weather",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "season_int",
        "month_sin",
        "month_cos",
    ]:
        if col in panel.columns:
            vals = pd.to_numeric(panel[col], errors="coerce")
            logger.info(
                "  %s: nunique=%d, mean=%.4f, zero%%=%.1f%%",
                col,
                vals.nunique(),
                vals.mean(),
                100 * (vals == 0).mean(),
            )
    logger.info("--- End Feature Summary ---")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger("train_temporal_model")

    logger.info("Starting temporal model training pipeline...")

    # Ensure output directories
    models_dir = OUTPUTS_DIR / "models"
    reports_dir = OUTPUTS_DIR / "reports"
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load cleaned data
    logger.info("Loading and cleaning raw data from %s", DATA_DIR)
    collision_data, ksi_data, road_network = load_and_clean_data(DATA_DIR)
    logger.info(
        "Loaded %d collision records, %d KSI records, %d road segments.",
        len(collision_data),
        len(ksi_data),
        len(road_network),
    )

    # 2) Event-level crash assignment
    logger.info("Performing event-level spatial join...")
    event_level = perform_spatial_join_event_level(
        collision_data=collision_data,
        ksi_data=ksi_data,
        road_network=road_network,
    )
    logger.info("Event-level dataset has %d rows.", len(event_level))

    # 3) Ensure stable segment_id and merge ADT/speed if available
    road_with_ids = _ensure_stable_segment_id(road_network)
    model_dataset = load_model_dataset(DATA_DIR)
    road_with_ids = merge_model_dataset_into_road_network(road_with_ids, model_dataset)

    # 3b) Load historical weather (city-wide Toronto) if available
    weather_data = load_historical_weather(DATA_DIR)

    # 3c) Merge TMC pedestrian/cyclist/vehicle volumes (intersections within 50m)
    tmc_data = load_tmc_data(DATA_DIR)
    road_with_ids = merge_tmc_into_road_network(road_with_ids, tmc_data)

    # 3d) Merge school zone flag (segments within 200m of any school)
    school_locations = load_school_locations(DATA_DIR)
    road_with_ids = merge_school_zones_into_road_network(
        road_with_ids, school_locations
    )

    # 3e) Merge TTC transit frequency (sum of avg trips/hour for stops within 150m)
    ttc_stops = load_ttc_gtfs(DATA_DIR)
    road_with_ids = merge_ttc_into_road_network(road_with_ids, ttc_stops)

    # 4) Build temporal configuration for *daily* predictive horizon.
    # W = 24 hours, H = 24 hours: model predicts next-day crash count.
    # Daily windows dramatically reduce zero-inflation vs hourly
    # (~99.85% zeros hourly → ~5-15% zeros daily) and give Stage 2 of the
    # hurdle model enough positive examples to train meaningfully.
    panel_config = PanelConfig(window_size_hours=24, horizon_hours=24)
    logger.info(
        "Building panel dataset with window_size=%dh, horizon=%dh (steps_ahead=%d)...",
        panel_config.window_size_hours,
        panel_config.horizon_hours,
        panel_config.steps_ahead(),
    )
    # 4a) Build a *sampled* daily training panel (positives + negatives).
    training_panel = build_weekly_sampled_future_panel(
        event_level_crashes=event_level,
        road_network=road_with_ids,
        weather_data=weather_data,
        window_size_hours=panel_config.window_size_hours,
        horizon_hours=panel_config.horizon_hours,
    )
    logger.info("Daily training panel shape: %s", training_panel.shape)

    # 4a-QA) Log feature summary so we can verify the model gets real variation.
    _log_panel_feature_summary(training_panel, logger)

    # 4b) Build inference panel: one row per segment for the latest window only.
    # Full segments×hours grid is intractable for hourly; API uses latest-window snapshot.
    full_panel = build_latest_window_inference_panel(
        event_level_crashes=event_level,
        road_network=road_with_ids,
        weather_data=weather_data,
        config=panel_config,
    )
    logger.info("Latest-window inference panel shape: %s", full_panel.shape)

    panel_path = reports_dir / "panel_latest.parquet"
    full_panel.to_parquet(panel_path, index=False)
    logger.info("Saved panel snapshot to %s", panel_path)

    # 5) Add tail-weighted training objective:
    #    w_final = w_sampling * w_tail(y), then normalize + cap for stability.
    training_panel = _add_tail_weighted_sample_weights(
        training_panel,
        target_col="future_crash_count",
        sampling_weight_col="sample_weight",
        output_weight_col="sample_weight_tail",
        alpha=2.0,
        tail_threshold=2.0,
        weight_cap=50.0,
    )

    # 6) Train hurdle model using future-looking labels and tail weights.
    # Stage 1 (binary): P(crash_occurs) — HistGBClassifier on full training set.
    # Stage 2 (count):  E[crash_count | crash] — HistGBRegressor(Poisson) on
    #                   positive windows only.  At daily resolution there are
    #                   enough positive windows for Stage 2 to train meaningfully.
    USE_HURDLE_MODEL = True
    if USE_HURDLE_MODEL:
        logger.info("Training HurdleTemporalTrainer (Stage 1 + Stage 2)...")
        trainer = HurdleTemporalTrainer(
            panel_config=panel_config,
            lambda_cap=50.0,
        )
    else:
        logger.info("Training TemporalCountModelTrainer (single-stage Poisson)...")
        trainer = TemporalCountModelTrainer(
            panel_config=panel_config,
            lambda_cap=50.0,
        )
    results = trainer.train_temporal_count_model(
        training_panel,
        target_col="future_crash_count",
        sample_weight_col="sample_weight_tail",
    )

    logger.info(
        "Temporal model metrics: MAE=%.4f, RMSE=%.4f, Poisson dev=%.4f",
        results["mae"],
        results["rmse"],
        results["poisson_deviance"],
    )

    # 7) Save the trained model
    model_path = models_dir / "toronto_temporal_count_model.pkl"
    trainer.save_model(str(model_path))
    logger.info("Saved temporal count model to %s", model_path)

    # 8) Save test-set results for diagnostics and outlier inspection
    diagnostics_path = reports_dir / "temporal_model_test_results.npz"
    test_sample_weight = (
        np.asarray(results["test_data_with_pred"]["sample_weight"])
        if "sample_weight" in results["test_data_with_pred"].columns
        else np.ones(len(results["y_test"]), dtype=float)
    )
    np.savez(
        diagnostics_path,
        y_test=np.asarray(results["y_test"]),
        y_pred=np.asarray(results["y_pred"]),
        sample_weight_test=test_sample_weight,
        mean_train_y=np.array(results["mean_train_y"]),
    )
    logger.info("Saved test results for diagnostics to %s", diagnostics_path)
    test_set_path = reports_dir / "temporal_model_test_set_with_pred.parquet"
    results["test_data_with_pred"].to_parquet(test_set_path, index=False)
    logger.info(
        "Saved test set with predictions for outlier inspection to %s", test_set_path
    )

    logger.info("Temporal model training pipeline completed successfully.")


if __name__ == "__main__":
    main()
