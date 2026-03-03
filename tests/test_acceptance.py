"""
High-level acceptance tests for the predictive crash-likelihood system.

These tests are designed to guard against:
- label leakage via current-window counts
- incorrect temporal splits (future leaking into train)
- inference on training rows instead of unseen future windows
- instability of segment IDs
- incorrect routing risk/time aggregation math
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd

from src.data_processing.spatial_join_fast import perform_spatial_join_event_level
from src.feature_engineering.panel_builder import (
    PanelConfig,
    build_panel_dataset,
    temporal_train_val_test_split,
)
from src.models.model_trainer import TemporalCountModelTrainer
from src.routing.road_graph import (
    build_road_graph,
    apply_risk_to_edge_costs,
    calculate_route_risk,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _dummy_data_for_tests():
    """
    Build a very small synthetic dataset to exercise the full stack without
    requiring the real Toronto data files.
    """
    # Simple road network with 2 segments and stable CENTRELINE_ID
    from shapely.geometry import LineString

    roads = gpd.GeoDataFrame(
        {
            "CENTRELINE_ID": [1, 2],
            "FROM_INTERSECTION_ID": [10, 20],
            "TO_INTERSECTION_ID": [20, 30],
            "ROAD_CLASS": ["arterial", "local"],
            "segment_length": [100.0, 80.0],
            "geometry": [
                LineString([(0.0, 0.0), (0.001, 0.0)]),
                LineString([(0.001, 0.0), (0.002, 0.0)]),
            ],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )

    # Minimal event-level crashes: one event on each segment in two windows
    base_time = pd.Timestamp("2020-01-01 00:00:00")
    events = gpd.GeoDataFrame(
        {
            "segment_id": [1, 1, 2],
            "event_datetime": [
                base_time,
                base_time + pd.Timedelta(hours=1),
                base_time + pd.Timedelta(hours=2),
            ],
            "crash_type": ["collision", "collision", "collision"],
            "is_ksi": [0, 0, 0],
            "fatalities": [0, 0, 0],
            "geometry": [
                roads.geometry.iloc[0].interpolate(0.5, normalized=True),
                roads.geometry.iloc[0].interpolate(0.5, normalized=True),
                roads.geometry.iloc[1].interpolate(0.5, normalized=True),
            ],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )

    return roads, events


def test_no_leakage_in_panel_features():
    """
    Verify that panel features do not include current-window crash counts
    as direct predictors of the future label.
    """
    roads, events = _dummy_data_for_tests()
    config = PanelConfig(window_size_hours=1, horizon_hours=2)
    panel = build_panel_dataset(events, roads, weather_data=None, config=config)

    # Future label must be shifted by steps_ahead rows, not raw hours
    steps = config.steps_ahead()
    # pick one segment and check that future_crash_count equals crash_count shifted by -steps
    seg = panel[panel["segment_id"] == 1].sort_values("window_start")
    shifted = seg["crash_count"].shift(-steps)
    assert np.allclose(
        seg["future_crash_count"].values, shifted.values, equal_nan=True
    ), "future_crash_count is not correctly shifted by steps_ahead."

    # Ensure current-window crash_count is not accidentally used as target
    # (i.e., they are not the same column)
    assert not np.array_equal(
        panel["crash_count"].values, panel["future_crash_count"].values
    ), "current-window crash_count must not be the prediction target."


def test_temporal_split_respects_time_ordering():
    """
    Ensure temporal_train_val_test_split uses ordered window_start values and
    that test windows are strictly after train windows.
    """
    roads, events = _dummy_data_for_tests()
    panel = build_panel_dataset(events, roads, weather_data=None, config=PanelConfig())

    train, val, test = temporal_train_val_test_split(panel)

    max_train_time = train["window_start"].max()
    min_test_time = test["window_start"].min()

    assert (
        min_test_time > max_train_time
    ), "Temporal split must ensure test windows are strictly after train windows."


def test_inference_on_unseen_future_windows():
    """
    Train a temporal count model and confirm inference can be performed on
    windows not present in the training set.
    """
    roads, events = _dummy_data_for_tests()
    panel = build_panel_dataset(events, roads, weather_data=None, config=PanelConfig())

    trainer = TemporalCountModelTrainer()
    _ = trainer.train_temporal_count_model(panel)

    # Build a "future" panel by shifting all windows forward in time
    future_panel = panel.copy()
    future_panel["window_start"] = future_panel["window_start"] + pd.Timedelta(days=30)
    X_future, _ = trainer.prepare_panel_features(future_panel)

    # Should not raise and should return an array of predictions
    lambda_future = trainer.predict_lambda(X_future)
    assert lambda_future.shape[0] == len(future_panel), "Inference must cover all future rows."


def test_stable_segment_ids():
    """
    Ensure that CENTRELINE_ID is used as the stable segment identifier and
    preserved through the graph construction.
    """
    from src.data_processing.spatial_join_fast import _ensure_stable_segment_id  # type: ignore

    roads, _ = _dummy_data_for_tests()
    roads_with_id = _ensure_stable_segment_id(roads)
    assert "segment_id" in roads_with_id.columns
    assert np.array_equal(
        roads_with_id["segment_id"].values, roads_with_id["CENTRELINE_ID"].values
    ), "segment_id must match CENTRELINE_ID."

    G = build_road_graph(roads_with_id)
    # Check that at least one edge carries the same segment_id as in the road network
    edge_segment_ids = {data["segment_id"] for _, _, data in G.edges(data=True)}
    assert 1 in edge_segment_ids and 2 in edge_segment_ids


def test_routing_risk_and_time_aggregation_math():
    """
    Verify that routing math aggregates λ * t correctly and converts to
    route-level probability with 1 - exp(-Σ(λ_i * t_i)).
    """
    roads, _ = _dummy_data_for_tests()
    roads = roads.copy()
    roads["segment_id"] = roads["CENTRELINE_ID"]

    G = build_road_graph(roads)

    # Assume simple λ_per_hour for two segments
    lam = {1: 0.1, 2: 0.2}  # crashes/hour
    apply_risk_to_edge_costs(G, lam, beta_hours_per_expected_crash=0.0)

    # Build path 10 -> 20 -> 30
    path = [10, 20, 30]
    summary = calculate_route_risk(G, path)

    # Manually compute expected crashes using stored travel_time_hours
    from src.routing.road_graph import path_edges  # type: ignore

    edges = path_edges(G, path)
    manual_expected = 0.0
    for _, _, data in edges:
        seg_id = data["segment_id"]
        t = data["travel_time_hours"]
        manual_expected += lam[seg_id] * t

    assert np.isclose(
        summary["expected_crashes"], manual_expected
    ), "Route expected crashes must equal Σ(λ_i * t_i)."

    manual_prob = 1.0 - np.exp(-manual_expected)
    assert np.isclose(
        summary["route_probability"], manual_prob
    ), "Route probability must be 1 - exp(-Σ(λ_i * t_i))."

