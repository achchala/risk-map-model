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
from pathlib import Path

import pandas as pd

from config import (
    DATA_DIR,
    OUTPUTS_DIR,
)

from src.data_processing.data_loader import load_and_clean_data  # type: ignore
from src.data_processing.spatial_join_fast import (  # type: ignore
    perform_spatial_join_event_level,
    _ensure_stable_segment_id,
)
from src.feature_engineering.panel_builder import (  # type: ignore
    PanelConfig,
    build_panel_dataset,
    build_weekly_sampled_future_panel,
)
from src.models.model_trainer import TemporalCountModelTrainer  # type: ignore


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

    # 3) Ensure stable segment_id on road network
    road_with_ids = _ensure_stable_segment_id(road_network)

    # 4) Build temporal configuration for a *predictive* weekly horizon.
    # We keep W = 1 week, H = 1 week to model "next-week" crash counts while
    # avoiding the full segments×days cartesian product.
    panel_config = PanelConfig(window_size_hours=24 * 7, horizon_hours=24 * 7)
    logger.info(
        "Building panel dataset with window_size=%dh, horizon=%dh (steps_ahead=%d)...",
        panel_config.window_size_hours,
        panel_config.horizon_hours,
        panel_config.steps_ahead(),
    )
    # 4a) Build a *sampled* weekly training panel with future-looking labels.
    training_panel = build_weekly_sampled_future_panel(
        event_level_crashes=event_level,
        road_network=road_with_ids,
        weather_data=None,  # weather integration can be added later
        window_size_hours=panel_config.window_size_hours,
        horizon_hours=panel_config.horizon_hours,
    )
    logger.info("Weekly training panel shape: %s", training_panel.shape)

    # 4b) Build a full panel snapshot for backend inference using the same
    # weekly configuration. This may be large but is written once and used
    # only for the latest window in the API.
    full_panel = build_panel_dataset(
        event_level_crashes=event_level,
        road_network=road_with_ids,
        weather_data=None,
        config=panel_config,
    )
    logger.info("Full weekly panel shape (snapshot): %s", full_panel.shape)

    panel_path = reports_dir / "panel_latest.parquet"
    full_panel.to_parquet(panel_path, index=False)
    logger.info("Saved panel snapshot to %s", panel_path)

    # 5) Train temporal count model using future-looking labels and sample weights
    logger.info("Training TemporalCountModelTrainer...")
    trainer = TemporalCountModelTrainer(panel_config=panel_config)
    results = trainer.train_temporal_count_model(
        training_panel,
        target_col="future_crash_count",
        sample_weight_col="sample_weight",
    )

    logger.info(
        "Temporal model metrics: MAE=%.4f, RMSE=%.4f, Poisson dev=%.4f",
        results["mae"],
        results["rmse"],
        results["poisson_deviance"],
    )

    # 6) Save the trained model
    model_path = models_dir / "toronto_temporal_count_model.pkl"
    trainer.save_model(str(model_path))
    logger.info("Saved temporal count model to %s", model_path)

    logger.info("Temporal model training pipeline completed successfully.")


if __name__ == "__main__":
    main()

