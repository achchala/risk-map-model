"""
Temporal panel dataset builder for crash likelihood modeling.

This module constructs a time-indexed panel over (segment_id, window_start)
with:
- crash counts per window
- static road attributes
- temporal indicators
- hooks for weather features
- past-only lag and rolling features
- future labels shifted by a configurable prediction horizon
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import geopandas as gpd

from pathlib import Path
import logging
import sys

# Add project root for config import
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import SEASON_MAPPING  # type: ignore

logger = logging.getLogger(__name__)


@dataclass
class PanelConfig:
    """Configuration for panel construction."""

    window_size_hours: int = 1
    horizon_hours: int = 6

    def steps_ahead(self) -> int:
        """
        Number of rows to shift for the future label.

        IMPORTANT:
        - steps_ahead = H / W (rows), NOT raw hours.
        - H must be divisible by W.
        """
        if self.horizon_hours % self.window_size_hours != 0:
            raise ValueError(
                f"Horizon {self.horizon_hours}h must be divisible by "
                f"window size {self.window_size_hours}h to avoid misaligned "
                f"future labels (hours vs rows bug)."
            )
        return self.horizon_hours // self.window_size_hours


def _compute_segment_centroids(road_network: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Compute centroid lat/lon for each segment for weather joins.
    """
    # Ensure we are in geographic CRS for lat/lon
    road_geo = road_network.to_crs("EPSG:4326")
    centroids = road_geo.geometry.centroid
    return pd.DataFrame(
        {
            "segment_id": road_network["segment_id"].values,
            "segment_centroid_lat": centroids.y.values,
            "segment_centroid_lon": centroids.x.values,
        }
    )


def _add_temporal_indicators(
    panel: pd.DataFrame, window_size_hours: Optional[int] = None
) -> pd.DataFrame:
    """
    Add temporal indicator features derived from window_start.

    For weekly (or coarser) windows, skips hour_of_day, day_of_week, is_weekend
    (they carry no real signal — each window spans 7 days). Keeps month and
    season which vary meaningfully.
    """
    panel = panel.copy()
    panel["month"] = panel["window_start"].dt.month
    panel["season"] = panel["month"].map(SEASON_MAPPING)
    if window_size_hours is None or not _is_weekly_or_coarser(window_size_hours):
        panel["hour_of_day"] = panel["window_start"].dt.hour
        panel["day_of_week"] = panel["window_start"].dt.dayofweek
        panel["is_weekend"] = panel["day_of_week"].isin([5, 6])
    return panel


def _encode_road_class_onehot(
    static: pd.DataFrame, road_network: gpd.GeoDataFrame
) -> pd.DataFrame:
    """
    Add one-hot encoded ROAD_CLASS columns for modeling.

    Keeps ROAD_CLASS for display; adds road_class_<value> columns (0/1) for each
    unique value in the road network. Uses full road_network to ensure all
    possible categories get columns, so inference sees the same schema.
    """
    static = static.copy()
    road_class_vals = (
        road_network["ROAD_CLASS"].fillna("unknown").astype(str).unique().tolist()
    )

    for val in road_class_vals:
        safe_name = str(val).replace(" ", "_").replace("None", "unknown")
        col_name = f"road_class_{safe_name}"
        static[col_name] = (
            static["ROAD_CLASS"].fillna("unknown").astype(str) == val
        ).astype(int)

    return static


def _attach_weather_features(
    panel: pd.DataFrame,
    weather_data: Optional[pd.DataFrame],
    grid_size: float = 0.01,
) -> pd.DataFrame:
    """
    Attach weather features to the panel using a simple grid-based join.

    Join keys:
    - datetime_hour = window_start floored to hour
    - (lat_grid, lon_grid) computed from segment centroids

    The function is written so that it is safe to call with weather_data=None;
    in that case it simply adds an is_missing_weather flag and returns.
    """
    panel = panel.copy()
    panel["datetime_hour"] = panel["window_start"].dt.floor("H")

    if weather_data is None or weather_data.empty:
        logger.warning(
            "Weather data is missing or empty; skipping weather features."
        )
        return panel

    if "datetime_hour" not in weather_data.columns:
        logger.warning("Weather data missing datetime_hour; skipping weather join.")
        panel["is_missing_weather"] = True
        return panel

    # City-wide weather (no lat/lon): join only on datetime_hour
    # Gridded weather: join on datetime_hour, lat_grid, lon_grid
    has_grid = "lat_grid" in weather_data.columns and "lon_grid" in weather_data.columns
    if has_grid:
        panel["lat_grid"] = (panel["segment_centroid_lat"] // grid_size) * grid_size
        panel["lon_grid"] = (panel["segment_centroid_lon"] // grid_size) * grid_size
        merge_on = ["datetime_hour", "lat_grid", "lon_grid"]
        wx_subset = weather_data
    else:
        merge_on = ["datetime_hour"]
        wx_subset = weather_data.drop(
            columns=[c for c in weather_data.columns if c in {"lat_grid", "lon_grid"}],
            errors="ignore",
        )

    merged = panel.merge(
        wx_subset,
        on=merge_on,
        how="left",
        suffixes=("", "_wx"),
    )

    weather_cols = [
        c
        for c in merged.columns
        if c
        in {
            "temperature",
            "precipitation",
            "visibility",
            "wind_speed",
            "weather_condition",
            "snow_mm",
        }
    ]

    merged["is_missing_weather"] = merged[weather_cols].isna().all(axis=1)

    # For numeric weather features, forward-fill within each segment_id
    numeric_weather = [c for c in weather_cols if merged[c].dtype != "object"]
    if numeric_weather:
        merged[numeric_weather] = (
            merged.groupby("segment_id")[numeric_weather].ffill()
        )

    return merged


def _is_weekly_or_coarser(window_size_hours: int) -> bool:
    """True when window is weekly (168h) or coarser."""
    return window_size_hours >= 24 * 6


def _compute_lag_features_from_sparse(
    panel: pd.DataFrame,
    crash_counts_sparse: pd.DataFrame,
    window_size_hours: int,
) -> pd.DataFrame:
    """
    Add past-only lag and rolling features by joining against a sparse
    crash-count lookup table keyed by (segment_id, window_start).

    For weekly (or coarser) windows: uses properly spaced lags to avoid
    redundancy — crashes_1_week_ago, crashes_2_weeks_ago, crashes_4_weeks_ago,
    rolling_mean_4_weeks, rolling_max_4_weeks.

    For finer windows: uses hour-based lags (past_crash_count_1h, 24h, 7d, etc).
    """
    panel = panel.copy()
    delta = pd.Timedelta(hours=window_size_hours)

    sparse = crash_counts_sparse[["segment_id", "window_start", "crash_count"]].drop_duplicates(
        subset=["segment_id", "window_start"]
    )

    if _is_weekly_or_coarser(window_size_hours):
        # Weekly-spaced lags: 1, 2, 4 windows back (distinct signal)
        lag_steps = [1, 2, 4]
        rolling_window = 4
        lag_col_names = [
            "crashes_1_week_ago",
            "crashes_2_weeks_ago",
            "crashes_4_weeks_ago",
        ]
        rolling_mean_name = "rolling_mean_4_weeks"
        rolling_max_name = "rolling_max_4_weeks"
    else:
        # Finer windows: hour-based. Use 24h rolling (not 30d) to avoid OOM.
        steps_24h = max(1, int(round(24 / window_size_hours)))
        steps_7d = max(1, int(round(24 * 7 / window_size_hours)))
        lag_steps = [1, steps_24h, steps_7d]
        rolling_window = min(24, steps_7d)  # Cap at 24 for memory
        lag_col_names = [
            "past_crash_count_1h",
            "past_crash_count_24h",
            "past_crash_count_7d",
        ]
        rolling_mean_name = "rolling_mean_24h"
        rolling_max_name = "rolling_max_24h"
        # Only compute lags we need (1..rolling_window + steps_7d), not 1..720.
        lag_steps_to_compute = sorted(
            set(range(1, rolling_window + 1)) | set(lag_steps)
        )

    max_lag = max(lag_steps + [rolling_window]) if _is_weekly_or_coarser(window_size_hours) else max(lag_steps_to_compute)

    # Memory-efficient path for fine windows: merge panel with each shifted sparse
    # one at a time instead of building a huge lag_table (avoids OOM).
    if _is_weekly_or_coarser(window_size_hours):
        lag_frames = {}
        for k in range(1, max_lag + 1):
            shifted = sparse.copy()
            shifted["window_start"] = shifted["window_start"] + k * delta
            lag_frames[k] = shifted.rename(columns={"crash_count": f"_lag_{k}"})
        lag_table = lag_frames[1][["segment_id", "window_start", "_lag_1"]]
        for k in range(2, max_lag + 1):
            lag_table = lag_table.merge(
                lag_frames[k][["segment_id", "window_start", f"_lag_{k}"]],
                on=["segment_id", "window_start"],
                how="outer",
            )
        lag_table[[f"_lag_{k}" for k in range(1, max_lag + 1)]] = lag_table[
            [f"_lag_{k}" for k in range(1, max_lag + 1)]
        ].fillna(0)
        for step, col_name in zip(lag_steps, lag_col_names):
            lag_table[col_name] = lag_table[f"_lag_{step}"]
        cols_rolling = [f"_lag_{k}" for k in range(1, rolling_window + 1)]
        lag_table[rolling_mean_name] = lag_table[cols_rolling].mean(axis=1)
        lag_table[rolling_max_name] = lag_table[cols_rolling].max(axis=1)
        feature_names = (
            ["segment_id", "window_start"]
            + lag_col_names
            + [rolling_mean_name, rolling_max_name]
        )
        lag_table = lag_table[feature_names]
        panel = panel.merge(lag_table, on=["segment_id", "window_start"], how="left")
    else:
        # Merge one lag at a time (only the steps we need) to avoid OOM.
        for k in lag_steps_to_compute:
            shifted = sparse[["segment_id", "window_start", "crash_count"]].copy()
            shifted["window_start"] = shifted["window_start"] + k * delta
            shifted = shifted.rename(columns={"crash_count": f"_lag_{k}"})
            panel = panel.merge(
                shifted, on=["segment_id", "window_start"], how="left"
            )
            panel[f"_lag_{k}"] = panel[f"_lag_{k}"].fillna(0)
        for step, col_name in zip(lag_steps, lag_col_names):
            panel[col_name] = panel[f"_lag_{step}"]
        cols_rolling = [f"_lag_{k}" for k in range(1, rolling_window + 1) if f"_lag_{k}" in panel.columns]
        if cols_rolling:
            panel[rolling_mean_name] = panel[cols_rolling].mean(axis=1)
            panel[rolling_max_name] = panel[cols_rolling].max(axis=1)
        else:
            panel[rolling_mean_name] = 0.0
            panel[rolling_max_name] = 0.0
        for k in lag_steps_to_compute:
            panel.drop(columns=[f"_lag_{k}"], inplace=True)

    for c in lag_col_names + [rolling_mean_name, rolling_max_name]:
        if c in panel.columns:
            panel[c] = panel[c].fillna(0)

    return panel


def build_panel_dataset(
    event_level_crashes: gpd.GeoDataFrame,
    road_network: gpd.GeoDataFrame,
    weather_data: Optional[pd.DataFrame] = None,
    config: Optional[PanelConfig] = None,
) -> pd.DataFrame:
    """
    Build a temporal panel indexed by (segment_id, window_start).

    - Aggregates event-level crashes into fixed windows
    - Fills the full cartesian product of segments × windows (zeroes where no crashes)
    - Adds static road features and temporal indicators
    - Optionally joins weather data
    - Adds past-only lag and rolling features
    - Adds future labels shifted by steps_ahead rows (H/W), not raw hours
    """
    if config is None:
        config = PanelConfig()

    logger.info(
        "Building panel dataset with window_size=%dh, horizon=%dh",
        config.window_size_hours,
        config.horizon_hours,
    )

    if event_level_crashes.empty:
        raise ValueError("event_level_crashes is empty; cannot build panel dataset.")

    # Ensure we have segment_id and event_datetime
    required_cols = {"segment_id", "event_datetime"}
    if not required_cols.issubset(event_level_crashes.columns):
        missing = required_cols - set(event_level_crashes.columns)
        raise ValueError(f"event_level_crashes missing required columns: {missing}")

    # 1. Create time windows covering the full range of events
    min_ts = event_level_crashes["event_datetime"].min().floor("H")
    max_ts = event_level_crashes["event_datetime"].max().ceil("H")

    window_starts = pd.date_range(
        min_ts,
        max_ts,
        freq=f"{config.window_size_hours}H",
    )

    logger.info(
        "Panel time range: %s → %s (%d windows)",
        min_ts,
        max_ts,
        len(window_starts),
    )

    # 2. Aggregate crashes per (segment_id, window_start)
    # origin=min_ts ensures Grouper bin edges align with the date_range grid
    # so the left-join in step 3 actually matches.
    grp_keys = [
        "segment_id",
        pd.Grouper(
            key="event_datetime",
            freq=f"{config.window_size_hours}H",
            label="left",
            origin=min_ts,
        ),
    ]
    agg_df = event_level_crashes.groupby(grp_keys).agg(
        is_ksi=("is_ksi", "sum"),
        fatalities=("fatalities", "sum"),
    )
    agg_df["crash_count"] = event_level_crashes.groupby(grp_keys).size()
    grouped = agg_df.reset_index().rename(columns={"event_datetime": "window_start"})

    # 3. Build full panel (all segments × all windows), filling missing with zeros.
    # To avoid an enormous cartesian product, restrict to segments that actually
    # appear in the event-level data.
    active_segments = event_level_crashes["segment_id"].unique()
    all_segments = np.intersect1d(active_segments, road_network["segment_id"].unique())
    full_panel_index = pd.MultiIndex.from_product(
        [all_segments, window_starts],
        names=["segment_id", "window_start"],
    )
    full_panel = full_panel_index.to_frame(index=False)

    panel = full_panel.merge(
        grouped,
        on=["segment_id", "window_start"],
        how="left",
    )

    for col in ["crash_count", "is_ksi", "fatalities"]:
        if col in panel.columns:
            panel[col] = panel[col].fillna(0)
        else:
            panel[col] = 0

    # 4. Attach static road features and centroids
    static_cols = ["segment_id", "segment_length", "ROAD_CLASS", "is_oneway", "from_intersection_degree", "to_intersection_degree"]
    for opt_col in ["FROM_INTERSECTION_ID", "TO_INTERSECTION_ID"]:
        if opt_col in road_network.columns:
            static_cols.append(opt_col)
    for adt_col in ["avg_daily_vol", "avg_speed", "avg_85th_percentile_speed", "speed_variance", "exposure"]:
        if adt_col in road_network.columns:
            static_cols.append(adt_col)

    static = road_network[static_cols].drop_duplicates("segment_id")
    static = _encode_road_class_onehot(static, road_network)
    centroids = _compute_segment_centroids(road_network)

    panel = panel.merge(static, on="segment_id", how="left")
    panel = panel.merge(centroids, on="segment_id", how="left")

    # 5. Temporal indicators
    panel["window_start"] = pd.to_datetime(panel["window_start"])
    panel = _add_temporal_indicators(panel, config.window_size_hours)

    # 6. Optional weather features
    panel = _attach_weather_features(panel, weather_data=weather_data)

    # 7. Past-only lag and rolling features (CRITICAL: shift first)
    panel = panel.sort_values(["segment_id", "window_start"])
    past_series = panel.groupby("segment_id")["crash_count"].shift(1)

    if _is_weekly_or_coarser(config.window_size_hours):
        # Weekly-spaced lags: 1, 2, 4 windows back (distinct signal)
        panel["crashes_1_week_ago"] = panel.groupby("segment_id")["crash_count"].shift(1)
        panel["crashes_2_weeks_ago"] = panel.groupby("segment_id")["crash_count"].shift(2)
        panel["crashes_4_weeks_ago"] = panel.groupby("segment_id")["crash_count"].shift(4)
        rolling_window = 4
        rolling_mean_name = "rolling_mean_4_weeks"
        rolling_max_name = "rolling_max_4_weeks"
    else:
        steps_24h = max(1, int(round(24 / config.window_size_hours)))
        steps_7d = max(1, int(round((24 * 7) / config.window_size_hours)))
        window_30d = max(1, int(round((24 * 30) / config.window_size_hours)))
        panel["past_crash_count_1h"] = panel.groupby("segment_id")["crash_count"].shift(1)
        panel["past_crash_count_24h"] = panel.groupby("segment_id")["crash_count"].shift(
            steps_24h
        )
        panel["past_crash_count_7d"] = panel.groupby("segment_id")["crash_count"].shift(
            steps_7d
        )
        rolling_window = max(steps_7d, window_30d)
        rolling_mean_name = "rolling_mean_7d"
        rolling_max_name = "rolling_max_30d"

    rolling_mean = (
        past_series.groupby(panel["segment_id"])
        .rolling(rolling_window, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    rolling_max = (
        past_series.groupby(panel["segment_id"])
        .rolling(rolling_window, min_periods=1)
        .max()
        .reset_index(level=0, drop=True)
    )
    panel[rolling_mean_name] = rolling_mean
    panel[rolling_max_name] = rolling_max

    # 8. Future labels using correct steps_ahead (H/W)
    k = config.steps_ahead()
    panel["future_crash_count"] = (
        panel.groupby("segment_id")["crash_count"].shift(-k)
    )

    # Drop rows where the future label is NaN (end of history per segment)
    before_drop = len(panel)
    panel = panel.dropna(subset=["future_crash_count"])
    logger.info(
        "Panel rows after dropping tail windows without future labels: %d (dropped %d).",
        len(panel),
        before_drop - len(panel),
    )

    return panel


def build_weekly_sampled_future_panel(
    event_level_crashes: gpd.GeoDataFrame,
    road_network: gpd.GeoDataFrame,
    weather_data: Optional[pd.DataFrame] = None,
    window_size_hours: int = 24 * 7,
    horizon_hours: int = 24 * 7,
    negative_multiplier: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Build a *sampled* weekly panel with a future-looking label using a join-based
    construction, without materialising the full segments×windows grid.

    Steps (per external design guidance):
    - Step 1: Aggregate crashes into sparse (segment_id, window_start) counts.
    - Step 2: Create training index of positives (crash_count>0) + sampled zeros.
    - Step 3: Attach static + temporal + optional weather features at time t.
    - Step 4: Define the label as crash_count(s, t+H) via a join on future_window_start.

    Returns a DataFrame with at least:
    - segment_id
    - window_start
    - crash_count (current window)
    - future_crash_count (label window)
    - sample_weight (for handling zero under-sampling)
    - feature columns (road attributes, temporal indicators, optional weather)
    """
    if event_level_crashes.empty:
        raise ValueError("event_level_crashes is empty; cannot build training panel.")

    required_cols = {"segment_id", "event_datetime"}
    if not required_cols.issubset(event_level_crashes.columns):
        missing = required_cols - set(event_level_crashes.columns)
        raise ValueError(f"event_level_crashes missing required columns: {missing}")

    logger.info(
        "Building weekly sampled training panel with window_size=%dh, horizon=%dh",
        window_size_hours,
        horizon_hours,
    )

    min_event_ts = event_level_crashes["event_datetime"].min().floor("H")

    # Step 1: Sparse crash counts per (segment_id, window_start)
    # origin=min_event_ts keeps bin edges consistent with build_panel_dataset.
    crash_counts = (
        event_level_crashes.groupby(
            [
                "segment_id",
                pd.Grouper(
                    key="event_datetime",
                    freq=f"{window_size_hours}H",
                    label="left",
                    origin=min_event_ts,
                ),
            ]
        )
        .size()
        .reset_index(name="crash_count")
        .rename(columns={"event_datetime": "window_start"})
    )

    # Ensure datetime and sort for reproducibility
    crash_counts["window_start"] = pd.to_datetime(crash_counts["window_start"])

    # Positives: all windows with at least one crash
    positives = crash_counts[crash_counts["crash_count"] > 0].copy()
    if positives.empty:
        raise ValueError(
            "No positive crash windows found when building weekly sampled panel."
        )

    n_pos = len(positives)
    logger.info("Weekly sampled panel: %d positive (segment, window) pairs.", n_pos)

    # Global time range for sampling windows
    min_ws = crash_counts["window_start"].min()
    max_ws = crash_counts["window_start"].max()
    all_windows = pd.date_range(
        min_ws,
        max_ws,
        freq=f"{window_size_hours}H",
    )

    # Active segments only (those that appear in the event-level data)
    active_segments = np.sort(event_level_crashes["segment_id"].unique())

    # Approximate number of true zero windows for the future label universe
    # (used only for weighting). Universe is active_segments × all_windows.
    approx_total_pairs = int(len(active_segments) * len(all_windows))
    approx_nonzero_pairs = n_pos
    approx_zero_pairs = max(approx_total_pairs - approx_nonzero_pairs, 1)

    # Step 2: Sample negative windows (0-crash windows) without building full grid
    rng = np.random.default_rng(random_state)
    target_neg = negative_multiplier * n_pos

    # MultiIndex of positive (segment_id, window_start) for exclusion
    pos_index = pd.MultiIndex.from_frame(
        positives[["segment_id", "window_start"]]
    )

    # Over-sample then filter to avoid while-loops
    max_trials = target_neg * 2
    seg_samples = rng.choice(active_segments, size=max_trials, replace=True)
    win_samples = rng.choice(all_windows, size=max_trials, replace=True)

    neg_candidates = pd.DataFrame(
        {
            "segment_id": seg_samples,
            "window_start": win_samples,
        }
    ).drop_duplicates()

    neg_index = pd.MultiIndex.from_frame(
        neg_candidates[["segment_id", "window_start"]]
    )
    mask_not_pos = ~neg_index.isin(pos_index)
    negatives = neg_candidates[mask_not_pos].head(target_neg).copy()
    negatives["crash_count"] = 0

    n_neg = len(negatives)
    logger.info(
        "Weekly sampled panel: %d sampled zero windows (requested up to %d).",
        n_neg,
        target_neg,
    )

    # Combine positives and sampled negatives into training index
    train_idx = pd.concat(
        [positives[["segment_id", "window_start", "crash_count"]], negatives],
        ignore_index=True,
    )

    # Step 3: Attach static and temporal features at time t
    static_cols = ["segment_id", "segment_length", "ROAD_CLASS", "is_oneway", "from_intersection_degree", "to_intersection_degree"]
    for opt_col in ["FROM_INTERSECTION_ID", "TO_INTERSECTION_ID"]:
        if opt_col in road_network.columns:
            static_cols.append(opt_col)
    for adt_col in ["avg_daily_vol", "avg_speed", "avg_85th_percentile_speed", "speed_variance", "exposure"]:
        if adt_col in road_network.columns:
            static_cols.append(adt_col)

    static = road_network[static_cols].drop_duplicates("segment_id")
    static = _encode_road_class_onehot(static, road_network)
    centroids = _compute_segment_centroids(road_network)

    panel = train_idx.merge(static, on="segment_id", how="left")
    panel = panel.merge(centroids, on="segment_id", how="left")

    panel["window_start"] = pd.to_datetime(panel["window_start"])
    panel = _add_temporal_indicators(panel, window_size_hours)
    panel = _attach_weather_features(panel, weather_data=weather_data)

    # Step 3b: Lag and rolling features from sparse crash history
    panel = _compute_lag_features_from_sparse(
        panel, crash_counts, window_size_hours
    )

    # Step 4: Define future label via join on (segment_id, future_window_start)
    horizon_delta = pd.to_timedelta(horizon_hours, unit="H")
    panel["future_window_start"] = panel["window_start"] + horizon_delta

    future_counts = crash_counts.rename(
        columns={
            "window_start": "future_window_start",
            "crash_count": "future_crash_count",
        }
    )[["segment_id", "future_window_start", "future_crash_count"]]

    panel = panel.merge(
        future_counts,
        on=["segment_id", "future_window_start"],
        how="left",
    )
    panel["future_crash_count"] = panel["future_crash_count"].fillna(0)

    # Sample weights: up-weight zero-label rows to approximate full universe
    n_zero_sampled = int((panel["future_crash_count"] == 0).sum())
    if n_zero_sampled == 0:
        logger.warning(
            "No zero-label rows found in sampled panel; sample_weight will be all ones."
        )
        panel["sample_weight"] = 1.0
    else:
        w0 = approx_zero_pairs / n_zero_sampled
        panel["sample_weight"] = 1.0
        panel.loc[panel["future_crash_count"] == 0, "sample_weight"] = w0
        logger.info(
            "Approximate zero-universe=%d, sampled zeros=%d, weight per zero≈%.3f",
            approx_zero_pairs,
            n_zero_sampled,
            w0,
        )

    return panel


def build_sampled_future_panel(
    event_level_crashes: gpd.GeoDataFrame,
    road_network: gpd.GeoDataFrame,
    weather_data: Optional[pd.DataFrame] = None,
    window_size_hours: int = 24 * 7,
    horizon_hours: int = 24 * 7,
    negative_multiplier: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Build a sampled training panel with future-looking labels for any window size.
    Use window_size_hours=1, horizon_hours=1 for hourly; 168, 168 for weekly.

    Mirrors build_weekly_sampled_future_panel logic; kept as alias for clarity.
    """
    return build_weekly_sampled_future_panel(
        event_level_crashes=event_level_crashes,
        road_network=road_network,
        weather_data=weather_data,
        window_size_hours=window_size_hours,
        horizon_hours=horizon_hours,
        negative_multiplier=negative_multiplier,
        random_state=random_state,
    )


def build_latest_window_inference_panel(
    event_level_crashes: gpd.GeoDataFrame,
    road_network: gpd.GeoDataFrame,
    weather_data: Optional[pd.DataFrame] = None,
    window_size_hours: int = 1,
    config: Optional[PanelConfig] = None,
) -> pd.DataFrame:
    """
    Build an inference panel for only the latest window: one row per segment.
    Avoids the full segments×windows grid, which is intractable for hourly data.

    Returns a DataFrame with the same schema as build_panel_dataset, suitable
    for API inference. All road_network segments get a row (cold-start uses
    lag=0). Uses the most recent window_start from crash data.
    """
    if event_level_crashes.empty:
        raise ValueError("event_level_crashes is empty; cannot build inference panel.")

    required_cols = {"segment_id", "event_datetime"}
    if not required_cols.issubset(event_level_crashes.columns):
        missing = required_cols - set(event_level_crashes.columns)
        raise ValueError(f"event_level_crashes missing required columns: {missing}")

    if config is None:
        config = PanelConfig(window_size_hours=window_size_hours, horizon_hours=window_size_hours)

    min_event_ts = event_level_crashes["event_datetime"].min().floor("H")

    crash_counts = (
        event_level_crashes.groupby(
            [
                "segment_id",
                pd.Grouper(
                    key="event_datetime",
                    freq=f"{config.window_size_hours}H",
                    label="left",
                    origin=min_event_ts,
                ),
            ]
        )
        .size()
        .reset_index(name="crash_count")
        .rename(columns={"event_datetime": "window_start"})
    )
    crash_counts["window_start"] = pd.to_datetime(crash_counts["window_start"])

    latest_window = crash_counts["window_start"].max()
    all_segments = road_network["segment_id"].unique()

    inference_idx = pd.DataFrame(
        {"segment_id": all_segments, "window_start": latest_window}
    )
    inference_idx = inference_idx.merge(
        crash_counts[["segment_id", "window_start", "crash_count"]],
        on=["segment_id", "window_start"],
        how="left",
    )
    inference_idx["crash_count"] = inference_idx["crash_count"].fillna(0)

    static_cols = ["segment_id", "segment_length", "ROAD_CLASS"]
    for c in ["is_oneway", "from_intersection_degree", "to_intersection_degree",
              "FROM_INTERSECTION_ID", "TO_INTERSECTION_ID",
              "avg_daily_vol", "avg_speed", "avg_85th_percentile_speed", "speed_variance", "exposure"]:
        if c in road_network.columns:
            static_cols.append(c)
    static = road_network[static_cols].drop_duplicates("segment_id")
    static = _encode_road_class_onehot(static, road_network)
    centroids = _compute_segment_centroids(road_network)

    panel = inference_idx.merge(static, on="segment_id", how="left")
    panel = panel.merge(centroids, on="segment_id", how="left")
    panel["window_start"] = pd.to_datetime(panel["window_start"])
    panel = _add_temporal_indicators(panel, config.window_size_hours)
    panel = _attach_weather_features(panel, weather_data=weather_data)
    panel = _compute_lag_features_from_sparse(
        panel, crash_counts, config.window_size_hours
    )

    panel["future_crash_count"] = 0.0
    logger.info(
        "Built latest-window inference panel: %d segments, window %s",
        len(panel),
        latest_window,
    )
    return panel


def temporal_train_val_test_split(
    panel: pd.DataFrame,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    test_frac: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split panel into train/val/test sets based on *ordered* unique window_start values.

    This ensures:
    - Train windows are earliest
    - Validation windows are in the middle
    - Test windows are most recent (future-like)
    """
    if not np.isclose(train_frac + val_frac + test_frac, 1.0):
        raise ValueError("train_frac + val_frac + test_frac must sum to 1.0")

    unique_windows = np.sort(panel["window_start"].unique())
    n_windows = len(unique_windows)
    if n_windows < 3:
        raise ValueError("Not enough distinct windows to perform temporal split.")

    train_end = int(n_windows * train_frac)
    val_end = train_end + int(n_windows * val_frac)

    train_windows = unique_windows[:train_end]
    val_windows = unique_windows[train_end:val_end]
    test_windows = unique_windows[val_end:]

    train_mask = panel["window_start"].isin(train_windows)
    val_mask = panel["window_start"].isin(val_windows)
    test_mask = panel["window_start"].isin(test_windows)

    train_data = panel[train_mask].copy()
    val_data = panel[val_mask].copy()
    test_data = panel[test_mask].copy()

    logger.info(
        "Temporal split windows: train=%d, val=%d, test=%d",
        len(train_windows),
        len(val_windows),
        len(test_windows),
    )

    return train_data, val_data, test_data

