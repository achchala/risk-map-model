"""
flask API server
serves risk predictions from the trained model
"""

# Reduce chance of double-free in numpy/OpenBLAS/MKL (set before any numeric imports)
import os
import atexit
import threading
import time

# Ensure a usable temp dir (scipy/sklearn need it). Fixes "No usable temporary directory" on some systems.
_fallback_tmp = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".tmp"))
os.makedirs(_fallback_tmp, exist_ok=True)
# Always set TMPDIR/TMP/TEMP so scipy/sklearn use a known-writable dir
os.environ["TMPDIR"] = _fallback_tmp
os.environ["TMP"] = _fallback_tmp
os.environ["TEMP"] = _fallback_tmp

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import geopandas as gpd
import json
import networkx as nx
import numpy as np
import pandas as pd
from pathlib import Path
import re
from typing import Optional
import logging
import sys
from shapely.geometry import Point, box

# import existing modules
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# model & pipeline imports
from src.models.model_trainer import TemporalCountModelTrainer, HurdleTemporalTrainer  # type: ignore
from src.data_processing.data_loader import (
    load_and_clean_data,
    load_model_dataset,
    load_road_network,
    merge_model_dataset_into_road_network,
)
from src.data_processing.spatial_join_fast import (
    _ensure_stable_segment_id,
    perform_spatial_join_fast,
)  # type: ignore
from src.feature_engineering.panel_builder import PanelConfig  # type: ignore
from src.routing.road_graph import (  # type: ignore
    build_road_graph,
    apply_risk_to_edge_costs,
    find_fastest_route,
    find_safer_route,
    calculate_route_risk,
    build_node_geometry,
    snap_to_graph,
    path_edges,
)

# import config constants
try:
    from config import COLLISION_COLUMNS, KSI_COLUMNS
except ImportError:
    # Fallback if config import fails
    COLLISION_COLUMNS = {
        "latitude": "LAT_WGS84",
        "longitude": "LONG_WGS84",
        "date": "OCC_DATE",
        "time": "OCC_HOUR",
        "injury": "INJURY_COLLISIONS",
        "fatalities": "FATALITIES",
    }
    KSI_COLUMNS = {
        "latitude": "LATITUDE",
        "longitude": "LONGITUDE",
        "date": "DATE",
        "time": "TIME",
        "injury": "INJURY",
        "fatalities": "FATAL_NO",
    }

app = Flask(__name__)
CORS(app)

# initialize temporal count model (predictive crash likelihood λ)
# Pipeline saves HurdleTemporalTrainer; single-stage pipeline saves TemporalCountModelTrainer
TEMPORAL_MODEL_PATH = (
    PROJECT_ROOT / "outputs" / "models" / "toronto_temporal_count_model.pkl"
)
temporal_trainer = None

try:
    _trainer = HurdleTemporalTrainer()
    _trainer.load_model(str(TEMPORAL_MODEL_PATH))
    temporal_trainer = _trainer
    logging.info("Temporal count model (hurdle) loaded successfully")
except Exception as e:
    logging.warning(f"Failed to load hurdle model: {e}")
    try:
        _trainer = TemporalCountModelTrainer()
        _trainer.load_model(str(TEMPORAL_MODEL_PATH))
        temporal_trainer = _trainer
        logging.info("Temporal count model loaded successfully")
    except Exception as e2:
        logging.warning(f"Failed to load temporal count model: {e2}")
        temporal_trainer = None


def _is_temporal_model_loaded():
    """True if temporal_trainer is loaded and ready for inference (single-stage or hurdle)."""
    if temporal_trainer is None:
        return False
    if getattr(temporal_trainer, "model", None) is not None:
        return True
    if getattr(temporal_trainer, "stage1", None) is not None:
        return True
    return False


# Load road network (required for segments and routing)
road_network = None
road_graph = None
node_coords = None
panel_data = None
lambda_per_hour_latest = None
latest_window_start = None
# Percentile thresholds for mapping λ → risk_label (computed when lambda map is built)
_lambda_p70 = None
_lambda_p90 = None
# Cache for apply_risk: skip recompute when (beta, combined_mult) unchanged
_route_last_beta: Optional[float] = None
_route_last_combined_mult: Optional[float] = None

try:
    data_dir = PROJECT_ROOT / "data"
    cache_dir = PROJECT_ROOT / "outputs" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    road_cache_path = cache_dir / "road_network_with_crashes.parquet"
    force_refresh = os.environ.get("FORCE_REFRESH_ROAD_CACHE", "").lower() in ("1", "true", "yes")

    road_network = None
    # Load road network with crash history: use cache if available (spatial join is slow)
    if road_cache_path.exists() and not force_refresh:
        try:
            road_network = gpd.read_parquet(road_cache_path)
            n_with_crashes = (
                int((road_network["num_total_crashes"] > 0).sum())
                if "num_total_crashes" in road_network.columns
                else 0
            )
            logging.info(
                f"Loaded {len(road_network)} road segments from cache; {n_with_crashes} with crash history"
            )
        except Exception as cache_err:
            logging.warning(f"Cache load failed: {cache_err}. Rebuilding...")
            road_cache_path.unlink(missing_ok=True)
            road_network = None  # fall through to rebuild

    if road_network is None:
        try:
            collision_data, ksi_data, road_network = load_and_clean_data(data_dir)
            road_network = perform_spatial_join_fast(
                collision_data, ksi_data, road_network
            )
            n_with_crashes = (
                int((road_network["num_total_crashes"] > 0).sum())
                if "num_total_crashes" in road_network.columns
                else 0
            )
            road_network.to_parquet(road_cache_path, index=False)
            logging.info(
                f"Loaded {len(road_network)} road segments; {n_with_crashes} with crash history (cached)"
            )
        except Exception as join_err:
            logging.warning(
                f"Spatial join failed (crash data may be missing): {join_err}. "
                "Using road network without crash history."
            )
            road_network = load_road_network(data_dir)
            road_network["num_total_crashes"] = 0
            road_network["num_ksi_crashes"] = 0
            road_network["fatality_count"] = 0
            road_network = _ensure_stable_segment_id(road_network)
            logging.info(f"Loaded {len(road_network)} road segments (no crash history)")

    # Merge ADT/speed for on-demand inference (segment_id already set by spatial join or _ensure_stable_segment_id)
    model_dataset = load_model_dataset(data_dir)
    road_network = merge_model_dataset_into_road_network(road_network, model_dataset)

    road_graph = build_road_graph(road_network)
    node_coords = build_node_geometry(road_network)

    logging.info(
        "Routing graph initialized with %d nodes and %d edges.",
        road_graph.number_of_nodes() if road_graph is not None else 0,
        road_graph.number_of_edges() if road_graph is not None else 0,
    )
except Exception as e:
    logging.warning(f"Could not load road network: {e}")

# Load latest panel snapshot for temporal model inference (optional)
PANEL_PATHS = [
    PROJECT_ROOT / "outputs" / "reports" / "panel_latest.parquet",
]

for path in PANEL_PATHS:
    try:
        if path.exists():
            panel_data = pd.read_parquet(path)
            logging.info(f"Loaded panel data from {path} with shape {panel_data.shape}")
            break
    except Exception as e:
        logging.warning(f"Could not load panel data from {path}: {e}")
        continue


def _lambda_to_risk_label(lam: float) -> str:
    """
    Map λ (crashes per hour) to risk_label (low/medium/high) using percentile thresholds.

    Uses precomputed _lambda_p70, _lambda_p90 from the full λ distribution.
    """
    global _lambda_p70, _lambda_p90
    if _lambda_p70 is None or _lambda_p90 is None:
        return "low"  # fallback if thresholds not yet computed
    if lam <= _lambda_p70:
        return "low"
    if lam <= _lambda_p90:
        return "medium"
    return "high"


def _compute_lambda_map_for_latest_window():
    """
    Compute λ_per_hour for each segment in the most recent panel window.

    This is used to annotate routing edges with expected crashes and to derive
    risk_label for the iOS app (temporal model replaces classification model).
    """
    global lambda_per_hour_latest, latest_window_start, _lambda_p70, _lambda_p90

    if not _is_temporal_model_loaded():
        raise RuntimeError("Temporal count model is not loaded.")
    if panel_data is None or panel_data.empty:
        raise RuntimeError("Panel data is not loaded.")

    latest_window_start = panel_data["window_start"].max()
    current_slice = panel_data[panel_data["window_start"] == latest_window_start].copy()
    if current_slice.empty:
        raise RuntimeError("No panel rows found for the latest window_start.")

    # Prepare features and predict λ per window
    X_current, _ = temporal_trainer.prepare_panel_features(current_slice)  # type: ignore[arg-type]
    lambda_window = temporal_trainer.predict_lambda(X_current)  # type: ignore[arg-type]

    # Convert to crashes per hour based on panel config
    window_size_hours = temporal_trainer.panel_config.window_size_hours  # type: ignore[assignment]
    lambda_per_hour = lambda_window / float(window_size_hours)

    # Map segment_id -> λ_per_hour (normalize keys to int for consistent lookup with graph)
    segment_ids = current_slice["segment_id"].values
    lambda_per_hour_latest = {
        int(sid) if hasattr(sid, "__int__") else sid: float(lam)
        for sid, lam in zip(segment_ids, lambda_per_hour)
    }

    # Compute percentile thresholds for λ → risk_label mapping (low ≤ p70, medium ≤ p90, high > p90)
    lam_values = np.array(list(lambda_per_hour_latest.values()), dtype=float)
    _lambda_p70 = float(np.percentile(lam_values, 70))
    _lambda_p90 = float(np.percentile(lam_values, 90))
    logging.info(
        "Computed λ_per_hour for latest window %s for %d segments (p70=%.6f, p90=%.6f).",
        latest_window_start,
        len(lambda_per_hour_latest),
        _lambda_p70,
        _lambda_p90,
    )


# Compute lambda map once at startup so request handlers don't run heavy numpy/sklearn in request context
# (avoids potential double-free / teardown issues in some environments)
if (
    _is_temporal_model_loaded()
    and panel_data is not None
    and not getattr(panel_data, "empty", True)
):
    try:
        _compute_lambda_map_for_latest_window()
        logging.info("Lambda map computed at startup")
    except Exception as e:
        logging.warning(f"Could not compute lambda map at startup: {e}")


def _get_feature_importance_map():
    """
    Get feature importance from the loaded model for ranking risk drivers.
    Hurdle: uses stage2 (count regressor); single-stage: uses model.
    Returns dict of feature_name -> importance, or empty dict if unavailable.
    """
    if not _is_temporal_model_loaded() or temporal_trainer is None:
        return {}
    model = getattr(temporal_trainer, "model", None) or getattr(
        temporal_trainer, "stage2", None
    )
    feature_columns = getattr(temporal_trainer, "feature_columns", None) or []
    if (
        model is not None
        and hasattr(model, "feature_importances_")
        and feature_columns
        and len(model.feature_importances_) == len(feature_columns)
    ):
        return dict(zip(feature_columns, model.feature_importances_.tolist()))
    return {}


def _get_risk_driver_features_for_segment(segment_id):
    """
    Extract risk driver features for a segment from the latest panel window.
    Uses temporal_trainer.feature_columns when available (model-aligned);
    otherwise falls back to a legacy key list.
    """
    if panel_data is None or panel_data.empty or latest_window_start is None:
        return {}

    row = panel_data[
        (panel_data["segment_id"] == segment_id)
        & (panel_data["window_start"] == latest_window_start)
    ]
    if row.empty:
        return {}

    row = row.iloc[0]

    # Derive keys from model's feature_columns when available
    feature_columns = getattr(temporal_trainer, "feature_columns", None) if temporal_trainer else None
    if feature_columns:
        keys = list(feature_columns)
    else:
        # Fallback when model not loaded
        keys = [
            "is_oneway",
            "from_intersection_degree",
            "to_intersection_degree",
            "segment_length",
            "day_of_week",
            "is_weekend",
            "month",
            "datetime_hour",
            "crashes_1d_ago",
            "crashes_7d_ago",
            "crashes_30d_ago",
            "rolling_mean_7d",
            "rolling_max_7d",
            "hist_crashes_per_year",
            "hist_crash_hour_ratio",
            "hist_crash_weekend_ratio",
            "temperature",
            "precipitation",
            "snow_depth_mm",
            "wind_speed",
            "is_freezing",
            "is_precip",
            "is_missing_weather",
        ]

    drivers = {}
    for k in keys:
        # Handle snow_mm alias (legacy panel column name)
        col = k if k in row.index else ("snow_mm" if k == "snow_depth_mm" else k)
        if col not in row.index:
            continue
        val = row[col]
        try:
            if isinstance(val, (np.integer, np.int64, np.int32)):
                v = float(int(val))
            elif isinstance(val, (np.floating, np.float64, np.float32)):
                v = float(val)
            elif isinstance(val, (int, float)):
                v = float(val)
            else:
                continue  # skip strings, bools - iOS expects [String: Double]
            if v == v:  # exclude NaN
                drivers[k] = v
        except (TypeError, ValueError):
            continue
    return drivers


_FEATURE_LABELS = {
    # Lag / history
    "crashes_1d_ago": "Crashes in last 24h",
    "crashes_7d_ago": "Crashes in last 7 days",
    "crashes_30d_ago": "Crashes in last 30 days",
    "rolling_mean_7d": "7-day rolling crash average",
    "rolling_max_7d": "7-day rolling crash peak",
    "rolling_mean_4_weeks": "4-week rolling crash average",
    "rolling_max_4_weeks": "4-week rolling crash peak",
    "rolling_mean_24h": "24h rolling crash average",
    "rolling_max_24h": "24h rolling crash peak",
    "rolling_max_30d": "30-day rolling crash peak",
    "crashes_1_week_ago": "Crashes 1 week ago",
    "crashes_2_weeks_ago": "Crashes 2 weeks ago",
    "crashes_4_weeks_ago": "Crashes 4 weeks ago",
    "past_crash_count_1h": "Crashes in last hour",
    "past_crash_count_24h": "Crashes in last 24h",
    "past_crash_count_7d": "Crashes in last 7 days",
    "hist_crashes_per_year": "Historical crashes per year",
    "hist_crash_hour_ratio": "Crash rate by hour pattern",
    "hist_crash_weekend_ratio": "Weekend crash rate",
    # Geometry
    "from_intersection_degree": "Intersection complexity (from)",
    "to_intersection_degree": "Intersection complexity (to)",
    "segment_length": "Segment length",
    "is_oneway": "One-way road",
    # Temporal
    "day_of_week": "Day of week",
    "is_weekend": "Weekend",
    "month": "Month",
    "datetime_hour": "Hour of day",
    "hour_of_day": "Hour of day",
    "month_sin": "Monthly seasonality",
    "month_cos": "Monthly seasonality",
    "season_int": "Season",
    "dow_sin": "Day-of-week pattern",
    "dow_cos": "Day-of-week pattern",
    # Traffic / ADT
    "avg_daily_vol": "Daily traffic volume",
    "avg_speed": "Average speed",
    "avg_85th_percentile_speed": "85th percentile speed",
    "avg_95th_percentile_speed": "95th percentile speed",
    "exposure": "Traffic exposure",
    "avg_wkdy_am_peak_vol": "AM peak volume",
    "avg_wkdy_pm_peak_vol": "PM peak volume",
    "avg_heavy_pct": "Heavy vehicle share",
    "log_volume": "Log traffic volume",
    # TMC exposure
    "tmc_daily_ped_vol": "Pedestrian volume",
    "tmc_daily_cyclist_vol": "Cyclist volume",
    "tmc_daily_vehicle_vol": "Vehicle volume (TMC)",
    # Context
    "is_school_zone": "School zone",
    "nearby_transit_frequency": "Transit frequency",
    "is_school_active_hour": "School active hours",
    # Weather
    "temperature": "Temperature",
    "precipitation": "Precipitation",
    "snow_depth_mm": "Snow depth",
    "snow_mm": "Snow depth",
    "wind_speed": "Wind speed",
    "is_freezing": "Freezing conditions",
    "is_precip": "Precipitation present",
    "is_missing_weather": "Missing weather data",
    "visibility": "Visibility",
    "weather_condition": "Weather condition",
}


def _road_class_label(col: str) -> str:
    """Convert road_class_X to human-readable label."""
    if col.startswith("road_class_"):
        name = col.replace("road_class_", "").replace("_", " ")
        return f"Road class: {name}"
    return col.replace("_", " ").title()


def _build_risk_explanation(
    drivers: dict,
    risk_label: str,
    *,
    num_total_crashes: Optional[int] = None,
    num_ksi_crashes: Optional[int] = None,
    fatality_count: Optional[int] = None,
) -> str:
    """
    Build human-readable paragraph explaining which factors contributed to risk.
    Includes crash history when available; ranks model factors by feature importance.
    """
    risk_text = risk_label.replace("_", " ").title()
    parts = []

    # Crash history: total crashes, KSI, fatalities (historical record on this segment)
    if (
        num_total_crashes is not None
        or num_ksi_crashes is not None
        or fatality_count is not None
    ):
        total = int(num_total_crashes or 0)
        ksi = int(num_ksi_crashes or 0)
        fatal = int(fatality_count or 0)
        if total > 0 or ksi > 0 or fatal > 0:
            hist_parts = []
            if total > 0:
                hist_parts.append(f"{total} crash{'es' if total != 1 else ''} on record")
            if ksi > 0:
                hist_parts.append(f"{ksi} serious injury (KSI)")
            if fatal > 0:
                hist_parts.append(f"{fatal} fatal{'ity' if fatal == 1 else 'ities'}")
            parts.append(
                f"Historical record: {', '.join(hist_parts)}."
            )

    hist_intro = " ".join(parts) if parts else ""
    factor_parts = []

    if not drivers and not hist_intro:
        return (
            f"This segment is rated {risk_text} risk based on the predicted crash rate (λ) "
            "for this segment."
        )

    importance_map = _get_feature_importance_map()

    def sort_key(item):
        k, v = item
        imp = importance_map.get(k, 0.0)
        if imp > 0:
            return -imp  # higher importance first
        if isinstance(v, (int, float)) and v == v:
            return -abs(v)  # fallback: magnitude
        return 0

    sorted_items = sorted(drivers.items(), key=sort_key)

    for k, v in sorted_items:
        if v is None or (isinstance(v, (int, float)) and v == 0):
            continue
        if k.startswith("road_class_") and isinstance(v, (int, float)) and v == 1:
            label = _road_class_label(k)
        else:
            label = _FEATURE_LABELS.get(k, _road_class_label(k) if k.startswith("road_class_") else k.replace("_", " ").title())
        if isinstance(v, (int, float)):
            if k.startswith("road_class_") and v == 1:
                factor_parts.append(label)
            elif "crash" in k.lower() or "hist" in k.lower():
                factor_parts.append(
                    f"{label} ({v:.1f})" if isinstance(v, float) else f"{label} ({v})"
                )
            elif "degree" in k:
                factor_parts.append(f"{label} ({v})")
            elif "ratio" in k:
                factor_parts.append(f"{label} ({v:.0%})" if v <= 1 else f"{label} ({v:.1f})")
            elif "length" in k:
                factor_parts.append(
                    f"{label} ({v:.1f}m)" if isinstance(v, float) else f"{label} ({v}m)"
                )
            elif "vol" in k or "volume" in k.lower() or "speed" in k.lower():
                factor_parts.append(
                    f"{label} ({v:,.0f})" if v >= 100 else f"{label} ({v:.1f})"
                )
            else:
                factor_parts.append(
                    f"{label} ({v:.1f})" if isinstance(v, float) else f"{label} ({v})"
                )
        else:
            factor_parts.append(f"{label} ({v})")
        if len(factor_parts) >= 5:
            break

    # Build final explanation: crash history first, then model factors
    intro = f"This segment is rated {risk_text} risk based on the predicted crash rate."
    if hist_intro:
        intro = f"{intro} {hist_intro}"
    if not factor_parts:
        return intro.rstrip(".")
    if len(factor_parts) == 1:
        factor_text = factor_parts[0]
    elif len(factor_parts) == 2:
        factor_text = f"{factor_parts[0]} and {factor_parts[1]}"
    else:
        factor_text = ", ".join(factor_parts[:-1]) + ", and " + factor_parts[-1]
    return f"{intro} The main contributing factors are: {factor_text}."


def _safe_int(val, default=0):
    """Convert value to int, handling None and NaN."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return default
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


def _safe_str_val(val, default=""):
    """Safe string conversion for segment location fields."""
    if val is None or (hasattr(val, "__float__") and np.isnan(val)):
        return default
    s = str(val).strip()
    return s if s else default


def _get_segment_location_description(row) -> str:
    """Build 'from X to Y' description using cross-street names or coordinates."""
    if row is None:
        return ""
    # Toronto Centreline: LF_NAME = left-facing street, RF_NAME = right-facing street (cross streets)
    lf = _safe_str_val(row.get("LF_NAME"), "")
    rf = _safe_str_val(row.get("RF_NAME"), "")
    if lf and rf:
        if lf == rf:
            return f"At {lf}"
        return f"From {lf} to {rf}"
    # Fallback: use first and last coordinates from geometry
    geom = row.get("geometry")
    if geom is not None:
        coords = []
        if hasattr(geom, "geoms") and geom.geoms:
            for g in geom.geoms:
                if hasattr(g, "coords"):
                    coords.extend(g.coords)
        elif hasattr(geom, "coords"):
            coords = list(geom.coords)
        if len(coords) >= 2:
            lon1, lat1 = coords[0][0], coords[0][1]
            lon2, lat2 = coords[-1][0], coords[-1][1]
            return f"From ({lat1:.4f}, {lon1:.4f}) to ({lat2:.4f}, {lon2:.4f})"
    return ""


@app.route("/api/health", methods=["GET"])
def health_check():
    """Health check endpoint. Use to see why risk endpoints may return 500."""
    model_ok = _is_temporal_model_loaded()
    panel_ok = panel_data is not None and not getattr(panel_data, "empty", True)
    segments_with_crashes = 0
    if road_network is not None and "num_total_crashes" in road_network.columns:
        segments_with_crashes = int((road_network["num_total_crashes"] > 0).sum())
    return jsonify(
        {
            "status": "healthy" if road_network is not None else "degraded",
            "temporal_model_loaded": model_ok,
            "road_network_loaded": road_network is not None,
            "routing_graph_built": road_graph is not None and bool(node_coords),
            "panel_loaded": panel_ok,
            "road_segments": len(road_network) if road_network is not None else 0,
            "segments_with_crash_history": segments_with_crashes,
            "panel_rows": int(len(panel_data)) if panel_ok else 0,
            "hint": (
                "Run train_temporal_model.py from project root and ensure outputs/models/toronto_temporal_count_model.pkl and outputs/reports/panel_latest.parquet exist."
                if (not model_ok or not panel_ok) and road_network is not None
                else None
            ),
        }
    )


@app.route("/api/risk-predictions", methods=["POST"])
def get_risk_predictions():
    """
    Get risk predictions for a geographic region using the temporal count model.

    Request body:
    {
        "north": float,
        "south": float,
        "east": float,
        "west": float
    }
    """
    if road_network is None:
        return jsonify({"error": "Road network not loaded"}), 500
    if not _is_temporal_model_loaded():
        return jsonify({"error": "Temporal model not loaded"}), 500
    if panel_data is None or panel_data.empty:
        return jsonify({"error": "Panel data not loaded"}), 500

    try:
        data = request.get_json()
        north = data.get("north")
        south = data.get("south")
        east = data.get("east")
        west = data.get("west")

        if lambda_per_hour_latest is None:
            _compute_lambda_map_for_latest_window()

        bbox = box(west, south, east, north)
        segments_in_bbox = road_network[road_network.geometry.intersects(bbox)].copy()

        def _get_risk_for_row(row):
            seg_id = row.get("segment_id") or row.get("CENTRELINE_ID", row.name)
            lam = lambda_per_hour_latest.get(seg_id, 0.0)  # type: ignore[union-attr]
            if lam == 0.0 and isinstance(seg_id, float) and not np.isnan(seg_id):
                lam = lambda_per_hour_latest.get(int(seg_id), 0.0)  # type: ignore[union-attr]
            return _lambda_to_risk_label(lam)

        segments_in_bbox["risk_label"] = segments_in_bbox.apply(
            _get_risk_for_row, axis=1
        )

        # Cap at 200 segments to avoid MapKit Metal buffer overflow (~50k resource limit)
        if len(segments_in_bbox) > 200:
            risk_priority = {"high": 3, "medium": 2, "low": 1}
            segments_in_bbox["_risk_priority"] = segments_in_bbox["risk_label"].map(
                risk_priority
            )
            segments_in_bbox = segments_in_bbox.sort_values(
                "_risk_priority", ascending=False
            ).head(200)

        results = []
        p70 = _lambda_p70 or 0.0
        p90 = _lambda_p90 or 0.0
        for idx, segment in segments_in_bbox.iterrows():
            coords = _extract_coordinates(segment.geometry)
            risk_label = segment["risk_label"]
            seg_id = segment.get("segment_id") or segment.get("CENTRELINE_ID", idx)
            lam = lambda_per_hour_latest.get(seg_id, 0.0)  # type: ignore[union-attr]
            if (
                lam == 0.0
                and isinstance(seg_id, (float, np.floating))
                and not np.isnan(seg_id)
            ):
                lam = lambda_per_hour_latest.get(int(seg_id), 0.0)  # type: ignore[union-attr]

            wh = temporal_trainer.panel_config.window_size_hours or 1  # type: ignore[union-attr]
            prob_crash = 1.0 - np.exp(-max(0, lam * wh))
            if risk_label == "high" and p90 > 0:
                confidence = 0.5 + 0.5 * min(1.0, (lam - p90) / (p90 * 0.5 + 1e-9))
            elif risk_label == "medium" and p70 < p90:
                confidence = 0.5 + 0.3 * min(
                    1.0, abs(lam - (p70 + p90) / 2) / (p90 - p70 + 1e-9)
                )
            elif risk_label == "low" and p70 > 0:
                confidence = 0.5 + 0.5 * min(1.0, (p70 - lam) / (p70 * 0.5 + 1e-9))
            else:
                confidence = max(float(prob_crash), 0.5)
            confidence = float(np.clip(confidence, 0.01, 1.0))

            drivers = _get_risk_driver_features_for_segment(seg_id)
            risk_explanation = _build_risk_explanation(
                drivers,
                risk_label,
                num_total_crashes=_safe_int(segment.get("num_total_crashes")),
                num_ksi_crashes=_safe_int(segment.get("num_ksi_crashes")),
                fatality_count=_safe_int(segment.get("fatality_count")),
            )

            segment_location = _get_segment_location_description(segment)
            result = {
                "id": str(seg_id),
                "LINEAR_NAME": segment.get("LINEAR_NAME", "Unknown"),
                "ROAD_CLASS": segment.get("ROAD_CLASS", "Unknown"),
                "segment_length": float(segment.get("segment_length", 0)),
                "segment_location": segment_location,
                "risk_label": risk_label,
                "confidence": confidence,
                "num_total_crashes": int(segment.get("num_total_crashes", 0)),
                "num_ksi_crashes": int(segment.get("num_ksi_crashes", 0)),
                "fatality_count": int(segment.get("fatality_count", 0)),
                "coordinates": coords[:50],
                "riskDrivers": drivers,
                "risk_explanation": risk_explanation,
            }
            results.append(result)

        logging.info(f"Returning {len(results)} segments for bbox")
        return jsonify(results)

    except Exception as e:
        logging.error(f"Error in risk predictions: {e}")
        import traceback

        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


def _route_risk_multipliers(weather_data, time_data):
    """
    Return (weather_mult, time_mult) for route risk adjustment.
    Each is >= 0; 1.0 means no change. Used when app sends live weather/time for safety-aware routing.
    """
    weather_mult = 1.0
    if weather_data and isinstance(weather_data, dict):
        cond = (weather_data.get("condition") or "clear").lower()
        if cond in ("rain", "heavy_rain"):
            weather_mult = 1.35
        elif cond in ("snow", "heavy_snow"):
            weather_mult = 1.5
        elif cond in ("fog", "mist"):
            weather_mult = 1.2
        elif cond == "thunderstorm":
            weather_mult = 1.5
        elif cond == "sleet":
            weather_mult = 1.4
    time_mult = 1.0
    if time_data and isinstance(time_data, dict):
        hour = time_data.get("hour")
        if hour is not None:
            try:
                h = int(hour)
                if h >= 23 or h < 5:
                    time_mult = 1.3
                elif (7 <= h <= 9) or (17 <= h <= 19):
                    time_mult = 1.25 if not time_data.get("is_weekend") else 1.1
            except (TypeError, ValueError):
                pass
    return (weather_mult, time_mult)


def _segment_row_by_id(seg_id):
    """Look up a road_network row by segment_id or CENTRELINE_ID. Returns None if not found."""
    if road_network is None:
        return None
    for col in ("segment_id", "CENTRELINE_ID"):
        if col not in road_network.columns:
            continue
        try:
            match = road_network[road_network[col].astype(type(seg_id)) == seg_id]
        except (TypeError, ValueError):
            match = road_network[road_network[col] == seg_id]
        if not match.empty:
            return match.iloc[0]
    return None


def _extract_coordinates(geometry):
    """Extract coordinates from geometry for JSON serialization"""
    coords = []
    try:
        # Handle multi-part geometries (MultiLineString, MultiPoint, etc.)
        # Check for 'geoms' attribute first to avoid trying .coords on multi-part geometries
        if hasattr(geometry, "geoms") and geometry.geoms:
            for geom in geometry.geoms:
                # Each sub-geometry should be a LineString, Point, etc.
                if hasattr(geom, "coords"):
                    for coord in geom.coords:
                        if len(coord) >= 2:
                            lon, lat = coord[0], coord[1]
                            coords.append({"latitude": lat, "longitude": lon})
        # Handle single-part geometries (LineString, Point, etc.)
        elif hasattr(geometry, "coords"):
            for coord in geometry.coords:
                if len(coord) >= 2:
                    lon, lat = coord[0], coord[1]
                    coords.append({"latitude": lat, "longitude": lon})
        # Handle Polygon - get exterior coordinates
        elif hasattr(geometry, "exterior"):
            return _extract_coordinates(geometry.exterior)
        # Handle Polygon with xy attribute
        elif hasattr(geometry, "xy"):
            x_coords, y_coords = geometry.xy
            for lon, lat in zip(x_coords, y_coords):
                coords.append({"latitude": lat, "longitude": lon})
    except (AttributeError, TypeError, IndexError) as e:
        # Suppress warnings for expected geometry structure differences
        # These are handled by the code above, no need to log
        pass
    except Exception as e:
        # Only log truly unexpected errors, and at debug level
        logging.debug(f"Error extracting coordinates: {e}")
    return coords


@app.route("/api/risk-prediction", methods=["POST"])
def get_risk_prediction():
    """
    Get risk prediction for a specific location using the temporal count model.

    Request body:
    {
        "latitude": float,
        "longitude": float
    }
    """
    if road_network is None:
        return jsonify({"error": "Road network not loaded"}), 500
    if not _is_temporal_model_loaded():
        return jsonify({"error": "Temporal model not loaded"}), 500
    if panel_data is None or panel_data.empty:
        return jsonify({"error": "Panel data not loaded"}), 500

    try:
        data = request.get_json()
        lat = data.get("latitude")
        lon = data.get("longitude")

        point = Point(lon, lat)

        nearest_idx = None
        min_distance = float("inf")

        for idx, segment in road_network.iterrows():
            distance = point.distance(segment.geometry)
            if distance < min_distance:
                min_distance = distance
                nearest_idx = idx

        if nearest_idx is not None and min_distance < 0.01:  # Within ~1km
            segment = road_network.loc[nearest_idx]

            if lambda_per_hour_latest is None:
                _compute_lambda_map_for_latest_window()

            seg_id = segment.get("segment_id") or segment.get(
                "CENTRELINE_ID", nearest_idx
            )
            lam = lambda_per_hour_latest.get(seg_id, 0.0)  # type: ignore[union-attr]
            if lam == 0.0 and isinstance(seg_id, float) and not np.isnan(seg_id):
                lam = lambda_per_hour_latest.get(int(seg_id), 0.0)  # type: ignore[union-attr]

            risk_label = _lambda_to_risk_label(lam)
            wh = temporal_trainer.panel_config.window_size_hours or 1  # type: ignore[union-attr]
            prob_crash = 1.0 - np.exp(-max(0, lam * wh))
            confidence = float(np.clip(prob_crash, 0.0, 1.0))

            if risk_label == "high":
                probabilities = {"low": 0.1, "medium": 0.1, "high": 0.8}
            elif risk_label == "medium":
                probabilities = {"low": 0.2, "medium": 0.7, "high": 0.1}
            else:
                probabilities = {"low": 0.8, "medium": 0.15, "high": 0.05}

            segment_info = {
                "id": str(segment.get("segment_id", nearest_idx)),
                "LINEAR_NAME": segment.get("LINEAR_NAME", "Unknown"),
                "ROAD_CLASS": segment.get("ROAD_CLASS", "Unknown"),
                "segment_length": float(segment.get("segment_length", 0)),
                "num_total_crashes": int(segment.get("num_total_crashes", 0)),
                "num_ksi_crashes": int(segment.get("num_ksi_crashes", 0)),
                "fatality_count": int(segment.get("fatality_count", 0)),
                "coordinates": _extract_coordinates(segment.geometry),
            }

            response = {
                "riskLevel": risk_label,
                "confidence": confidence,
                "probabilities": probabilities,
                "segmentInfo": segment_info,
            }
            return jsonify(response)

        response = {
            "riskLevel": "low",
            "confidence": 0.5,
            "probabilities": {"low": 0.8, "medium": 0.15, "high": 0.05},
            "segmentInfo": None,
        }
        return jsonify(response)

    except Exception as e:
        logging.error(f"Error in risk prediction: {e}")
        import traceback

        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/routes/safety-aware", methods=["POST"])
def get_safety_aware_route():
    """
    Compute fastest and safer routes between two points, using the temporal
    crash likelihood model and routing graph.

    Request body:
    {
        "origin": {"latitude": float, "longitude": float},
        "destination": {"latitude": float, "longitude": float},
        "beta": float (optional, risk-avoidance strength),
        "weather": {"condition": str, "temperature": float, ...} (optional, for risk adjustment),
        "time_of_day": {"hour": int, "is_weekend": bool} (optional, for risk adjustment)
    }
    """
    if (
        not _is_temporal_model_loaded()
        or road_graph is None
        or node_coords is None
        or not node_coords
    ):
        return (
            jsonify(
                {
                    "error": "Temporal model or routing graph not initialized. "
                    "Road network may lack intersection IDs. Check /api/health."
                }
            ),
            500,
        )

    try:
        data = request.get_json()
        origin = data.get("origin", {})
        destination = data.get("destination", {})
        beta = float(data.get("beta", 1.0))

        o_lat = origin.get("latitude")
        o_lon = origin.get("longitude")
        d_lat = destination.get("latitude")
        d_lon = destination.get("longitude")

        if None in (o_lat, o_lon, d_lat, d_lon):
            return (
                jsonify(
                    {
                        "error": "origin and destination must include latitude and longitude"
                    }
                ),
                400,
            )

        weather_data = data.get("weather")
        time_data = data.get("time_of_day")
        weather_mult, time_mult = _route_risk_multipliers(weather_data, time_data)
        combined_mult = weather_mult * time_mult

        origin_point = Point(o_lon, o_lat)
        dest_point = Point(d_lon, d_lat)

        # Ensure λ map is ready
        global lambda_per_hour_latest
        if lambda_per_hour_latest is None:
            _compute_lambda_map_for_latest_window()

        # Optionally adjust λ by current weather/time so safer route reflects conditions
        lam_for_routing = lambda_per_hour_latest
        if combined_mult != 1.0 and lambda_per_hour_latest:
            lam_for_routing = {
                k: v * combined_mult for k, v in lambda_per_hour_latest.items()
            }

        # Apply λ to edges. Skip if (beta, combined_mult) unchanged from last request.
        global _route_last_beta, _route_last_combined_mult
        if _route_last_beta != beta or _route_last_combined_mult != combined_mult:
            lam_values = list(lam_for_routing.values()) if lam_for_routing else []
            default_lam = float(np.median(lam_values)) if lam_values else 0.0
            apply_risk_to_edge_costs(
                road_graph,
                lam_for_routing,
                beta_hours_per_expected_crash=beta,
                default_lam_per_hour=default_lam,
            )  # type: ignore[arg-type]
            _route_last_beta = beta
            _route_last_combined_mult = combined_mult

        # Snap origin/destination to nearest graph nodes
        try:
            start_node = snap_to_graph(origin_point, node_coords)  # type: ignore[arg-type]
            end_node = snap_to_graph(dest_point, node_coords)  # type: ignore[arg-type]
        except ValueError as e:
            return (
                jsonify(
                    {
                        "error": str(e),
                        "hint": "Origin and destination must be within 300m of the road network (Toronto centreline).",
                    }
                ),
                400,
            )

        # Find fastest and safer paths
        fastest_path = find_fastest_route(road_graph, start_node, end_node)  # type: ignore[arg-type]
        safer_path = find_safer_route(road_graph, start_node, end_node)  # type: ignore[arg-type]

        fastest_summary = calculate_route_risk(road_graph, fastest_path)  # type: ignore[arg-type]
        safer_summary = calculate_route_risk(road_graph, safer_path)  # type: ignore[arg-type]

        # Collect segments along each path
        fastest_edges = path_edges(road_graph, fastest_path)  # type: ignore[arg-type]
        safer_edges = path_edges(road_graph, safer_path)  # type: ignore[arg-type]

        fastest_segments_set = {data["segment_id"] for _, _, data in fastest_edges}
        safer_segments_set = {data["segment_id"] for _, _, data in safer_edges}

        avoided_segments = sorted(fastest_segments_set - safer_segments_set)

        def _safe_str(val, default="Unknown"):
            if val is None or (hasattr(val, "__float__") and np.isnan(val)):
                return default
            return str(val)

        def _build_segment_list(edges):
            out = []
            for _, _, data in edges:
                seg_id = data["segment_id"]
                seg_id_int = int(seg_id) if hasattr(seg_id, "__int__") else seg_id
                lam = (
                    float(lambda_per_hour_latest.get(seg_id, 0.0))
                    if lambda_per_hour_latest
                    else 0.0
                )
                expected_crashes = float(data.get("expected_crashes", 0.0))
                risk_label = _lambda_to_risk_label(lam)
                row = _segment_row_by_id(seg_id)
                coords = (
                    _extract_coordinates(row.geometry)[:50] if row is not None else []
                )
                out.append(
                    {
                        "segmentId": seg_id_int,
                        "coordinates": coords,
                        "LINEAR_NAME": (
                            _safe_str(row.get("LINEAR_NAME"), "Unknown")
                            if row is not None
                            else "Unknown"
                        ),
                        "ROAD_CLASS": (
                            _safe_str(row.get("ROAD_CLASS"), "Unknown")
                            if row is not None
                            else "Unknown"
                        ),
                        "lambdaPerHour": lam,
                        "expectedCrashes": expected_crashes,
                        "risk_label": risk_label,
                    }
                )
            return out

        def _count_risk_labels(segments_list):
            high = sum(1 for s in segments_list if s.get("risk_label") == "high")
            medium = sum(1 for s in segments_list if s.get("risk_label") == "medium")
            low = sum(1 for s in segments_list if s.get("risk_label") == "low")
            return high, medium, low

        fastest_segments_list = _build_segment_list(fastest_edges)
        safer_segments_list = _build_segment_list(safer_edges)
        fastest_high, fastest_medium, fastest_low = _count_risk_labels(fastest_segments_list)
        safer_high, safer_medium, safer_low = _count_risk_labels(safer_segments_list)

        # Build risk driver explanations for avoided segments with geometry and labels
        avoided_details = []
        for seg_id in avoided_segments:
            lam = (
                float(lambda_per_hour_latest.get(seg_id, 0.0))
                if lambda_per_hour_latest is not None
                else 0.0
            )
            drivers = _get_risk_driver_features_for_segment(seg_id)
            risk_label = _lambda_to_risk_label(lam)
            row = _segment_row_by_id(seg_id)
            risk_explanation = _build_risk_explanation(
                drivers,
                risk_label,
                num_total_crashes=_safe_int(row.get("num_total_crashes")) if row is not None else 0,
                num_ksi_crashes=_safe_int(row.get("num_ksi_crashes")) if row is not None else 0,
                fatality_count=_safe_int(row.get("fatality_count")) if row is not None else 0,
            )
            coords = _extract_coordinates(row.geometry)[:50] if row is not None else []
            seg_id_int = int(seg_id) if hasattr(seg_id, "__int__") else seg_id
            segment_location = (
                _get_segment_location_description(row) if row is not None else ""
            )
            avoided_details.append(
                {
                    "segmentId": seg_id_int,
                    "lambdaPerHour": lam,
                    "riskDrivers": drivers,
                    "risk_explanation": risk_explanation,
                    "segment_location": segment_location,
                    "coordinates": coords,
                    "LINEAR_NAME": (
                        _safe_str(row.get("LINEAR_NAME"), "Unknown")
                        if row is not None
                        else "Unknown"
                    ),
                    "ROAD_CLASS": (
                        _safe_str(row.get("ROAD_CLASS"), "Unknown")
                        if row is not None
                        else "Unknown"
                    ),
                    "risk_label": risk_label,
                    "num_total_crashes": _safe_int(row.get("num_total_crashes")) if row is not None else 0,
                    "num_ksi_crashes": _safe_int(row.get("num_ksi_crashes")) if row is not None else 0,
                    "fatality_count": _safe_int(row.get("fatality_count")) if row is not None else 0,
                }
            )

        response = {
            "fastest": {
                "nodes": fastest_path,
                "segmentIds": list(fastest_segments_set),
                "segments": fastest_segments_list,
                "summary": {
                    "totalTravelTimeHours": float(
                        fastest_summary["total_travel_time_hours"]
                    ),
                    "expectedCrashes": float(fastest_summary["expected_crashes"]),
                    "routeProbability": float(fastest_summary["route_probability"]),
                    "highRiskSegments": fastest_high,
                    "mediumRiskSegments": fastest_medium,
                    "lowRiskSegments": fastest_low,
                },
            },
            "safer": {
                "nodes": safer_path,
                "segmentIds": list(safer_segments_set),
                "segments": safer_segments_list,
                "summary": {
                    "totalTravelTimeHours": float(
                        safer_summary["total_travel_time_hours"]
                    ),
                    "expectedCrashes": float(safer_summary["expected_crashes"]),
                    "routeProbability": float(safer_summary["route_probability"]),
                    "highRiskSegments": safer_high,
                    "mediumRiskSegments": safer_medium,
                    "lowRiskSegments": safer_low,
                },
            },
            "avoidedSegments": avoided_details,
            "betaHoursPerExpectedCrash": beta,
        }

        return jsonify(response)

    except Exception as e:
        logging.error(f"Error in safety-aware routing: {e}")
        import traceback

        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/debug/route-diversity", methods=["GET"])
def get_route_diversity():
    """
    Debug: sample OD pairs and report how often fastest != safer, and whether
    same-path cases are explained by having only one route (route structure).
    Uses fast connectivity check (remove fastest-path edges, test if alternate exists).
    """
    if road_graph is None or node_coords is None or lambda_per_hour_latest is None:
        return jsonify({"error": "Graph or lambda map not ready"}), 503

    import random

    # Apply risk to edges (same as route endpoint)
    lam_values = list(lambda_per_hour_latest.values()) if lambda_per_hour_latest else []
    default_lam = float(np.median(lam_values)) if lam_values else 0.0
    apply_risk_to_edge_costs(
        road_graph,
        lambda_per_hour_latest,
        beta_hours_per_expected_crash=1.0,
        default_lam_per_hour=default_lam,
    )

    nodes = list(road_graph.nodes())
    if len(nodes) < 2:
        return jsonify({"error": "Graph has too few nodes"}), 503

    n_samples = 30
    od_pairs = []
    seen = set()
    for _ in range(n_samples * 20):
        if len(od_pairs) >= n_samples:
            break
        u, v = random.sample(nodes, 2)
        if u == v or (u, v) in seen:
            continue
        try:
            nx.dijkstra_path_length(road_graph, u, v, weight="travel_time_hours")
            od_pairs.append((u, v))
            seen.add((u, v))
        except Exception:
            continue

    n_differ = 0
    n_same_single = 0
    n_same_multi = 0

    for start, end in od_pairs:
        fastest = find_fastest_route(road_graph, start, end)
        safer = find_safer_route(road_graph, start, end)
        fastest_segs = {
            road_graph[u][v]["segment_id"]
            for u, v in zip(fastest[:-1], fastest[1:])
        }
        safer_segs = {
            road_graph[u][v]["segment_id"]
            for u, v in zip(safer[:-1], safer[1:])
        }
        if fastest_segs != safer_segs:
            n_differ += 1
            continue
        # Fast check: remove fastest-path edges; if still connected, alternate exists
        edges_to_remove = list(zip(fastest[:-1], fastest[1:]))
        G_test = road_graph.copy()
        for u, v in edges_to_remove:
            if G_test.has_edge(u, v):
                G_test.remove_edge(u, v)
        has_alternate = nx.has_path(G_test, start, end)
        if has_alternate:
            n_same_multi += 1
        else:
            n_same_single += 1

    total = len(od_pairs)
    return jsonify({
        "n_od_pairs_tested": total,
        "paths_differ": n_differ,
        "pct_paths_differ": round(100 * n_differ / total, 1) if total else 0,
        "paths_same_single_route": n_same_single,
        "paths_same_multi_route": n_same_multi,
        "interpretation": (
            "When fastest=safest and single_route: route structure explains it (no alternatives). "
            "When fastest=safest and multi_route: beta or λ variation may be too small."
        ),
    })


@app.route("/api/debug/lambda-stats", methods=["GET"])
def get_lambda_stats():
    """
    Debug endpoint: return λ distribution from the latest lambda map.
    Use to check if any segments have λ = 0.
    """
    if lambda_per_hour_latest is None:
        return jsonify({"error": "Lambda map not computed"}), 503
    lam_values = np.array(list(lambda_per_hour_latest.values()), dtype=float)
    n_total = len(lam_values)
    n_zero = int((lam_values == 0).sum())
    n_near_zero = int((lam_values < 1e-10).sum())
    return jsonify({
        "n_segments": n_total,
        "n_exactly_zero": n_zero,
        "pct_exactly_zero": round(100 * n_zero / n_total, 2) if n_total else 0,
        "n_near_zero_lt_1e10": n_near_zero,
        "min": float(lam_values.min()),
        "max": float(lam_values.max()),
        "mean": float(lam_values.mean()),
        "median": float(np.median(lam_values)),
    })


def _google_duration_text_to_seconds(text: str) -> Optional[float]:
    normalized = (
        str(text)
        .lower()
        .replace("hours", "hr")
        .replace("hour", "hr")
        .replace("minutes", "min")
        .replace("minute", "min")
        .strip()
    )
    if not normalized:
        return None
    hours = 0
    minutes = 0
    hr_match = re.search(r"(\d+)\s*hr", normalized)
    min_match = re.search(r"(\d+)\s*min", normalized)
    if hr_match:
        hours = int(hr_match.group(1))
    if min_match:
        minutes = int(min_match.group(1))
    total_seconds = float(hours * 3600 + minutes * 60)
    return total_seconds if total_seconds > 0 else None


def _format_duration_seconds(seconds: float) -> str:
    total_minutes = int(round(seconds / 60.0))
    hours, minutes = divmod(total_minutes, 60)
    if hours > 0:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"


_google_playwright = None
_google_browser = None
_google_timeout_error = None
_google_scrape_lock = threading.Lock()
_GOOGLE_DURATION_RE = re.compile(r"^(?:(?:\d+\s*hr\s*)?\d+\s*min|\d+\s*hr)$", re.IGNORECASE)


def _shutdown_google_browser():
    global _google_browser, _google_playwright
    if _google_browser is not None:
        try:
            _google_browser.close()
        except Exception:
            pass
        _google_browser = None
    if _google_playwright is not None:
        try:
            _google_playwright.stop()
        except Exception:
            pass
        _google_playwright = None


def _ensure_google_browser():
    global _google_browser, _google_playwright, _google_timeout_error
    if _google_browser is not None and _google_playwright is not None and _google_timeout_error is not None:
        return _google_browser, _google_timeout_error

    try:
        from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency: playwright. Install with `pip install playwright` and `playwright install chromium`."
        ) from exc

    _google_playwright = sync_playwright().start()
    _google_browser = _google_playwright.chromium.launch(
        headless=True,
        args=["--disable-dev-shm-usage"],
    )
    _google_timeout_error = PlaywrightTimeoutError
    logging.info("[google-eta] Browser launched")
    return _google_browser, _google_timeout_error


atexit.register(_shutdown_google_browser)


def _block_heavy_google_resources(route):
    request = route.request
    if request.resource_type in {"image", "media", "font"}:
        route.abort()
        return
    route.continue_()


def _scrape_google_candidates(page):
    return page.evaluate(
        """
        () => {
          const durationRe = /^(?:(?:\\d+\\s*hr\\s*)?\\d+\\s*min|\\d+\\s*hr)$/i;
          const nodes = [];
          const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);

          while (walker.nextNode()) {
            const textNode = walker.currentNode;
            const text = (textNode.textContent || "").replace(/\\s+/g, " ").trim();
            if (!durationRe.test(text)) continue;

            const parent = textNode.parentElement;
            if (!parent) continue;

            const style = window.getComputedStyle(parent);
            if (style.visibility === "hidden" || style.display === "none") continue;

            const rect = parent.getBoundingClientRect();
            if (rect.width <= 0 || rect.height <= 0) continue;

            nodes.push({
              text,
              x: rect.x,
              y: rect.y,
              width: rect.width,
              height: rect.height,
              tag: parent.tagName,
              ariaLabel: parent.getAttribute("aria-label") || "",
            });
          }

          const deduped = [];
          const seen = new Set();
          for (const node of nodes) {
            const key = `${node.text}|${Math.round(node.x)}|${Math.round(node.y)}`;
            if (seen.has(key)) continue;
            seen.add(key);
            deduped.push(node);
          }

          deduped.sort((a, b) => {
            if (Math.abs(a.y - b.y) > 12) return a.y - b.y;
            return a.x - b.x;
          });

          return deduped;
        }
        """
    )


def _pick_best_google_candidate(candidates):
    enriched = []
    for item in candidates or []:
        text = str(item.get("text", "")).strip()
        if not _GOOGLE_DURATION_RE.match(text):
            continue
        minutes = 0
        hr_match = re.search(r"(\d+)\s*hr", text.lower())
        min_match = re.search(r"(\d+)\s*min", text.lower())
        if hr_match:
            minutes += int(hr_match.group(1)) * 60
        if min_match:
            minutes += int(min_match.group(1))
        enriched.append({**item, "minutes": minutes})

    if not enriched:
        return None

    directions_panel = [c for c in enriched if c["x"] < 700 and c["y"] < 900]
    return directions_panel[0] if directions_panel else enriched[0]


def _scrape_google_maps_eta_with_page(page, url: str, timeout_seconds: int = 10, label: str = "route") -> dict:
    logging.info("[google-eta] START %s scrape", label)
    try:
        page.goto(url, wait_until="domcontentloaded")
        page.wait_for_timeout(700)
    except _google_timeout_error:
        logging.warning("[google-eta] FAIL %s scrape: timed out waiting for page", label)
        raise RuntimeError("Timed out waiting for Google Maps to load")

    candidates = []
    deadline = time.monotonic() + min(float(timeout_seconds), 4.0)
    while time.monotonic() < deadline:
        candidates = _scrape_google_candidates(page)
        if candidates:
            break
        page.wait_for_timeout(250)

    best_guess = _pick_best_google_candidate(candidates)
    seconds = None
    if best_guess and best_guess.get("minutes") is not None:
        parsed_seconds = float(best_guess["minutes"]) * 60.0
        if parsed_seconds > 0:
            seconds = parsed_seconds
    if seconds is None:
        seconds = _google_duration_text_to_seconds((best_guess or {}).get("text", ""))

    if seconds is None:
        logging.warning(
            "[google-eta] FAIL %s scrape: no ETA found (candidate_count=%s)",
            label,
            len(candidates),
        )
        raise RuntimeError("No ETA found in Google Maps page")

    logging.info(
        "[google-eta] SUCCESS %s scrape: %s from '%s' (candidates=%s)",
        label,
        _format_duration_seconds(seconds),
        best_guess.get("text", "") if best_guess else "",
        len(candidates),
    )
    return {
        "seconds": seconds,
        "best_guess": best_guess,
        "candidate_count": len(candidates),
        "url": url,
    }


@app.route("/api/debug/google-maps-eta", methods=["POST"])
def debug_google_maps_eta():
    """
    Dev/debug endpoint that scrapes ETA text from Google Maps direction URLs.
    Expected JSON body:
      { "urls": { "fastest": "<url>", "safer": "<url>" } }
    """
    body = request.get_json(silent=True) or {}
    urls = body.get("urls")
    if not isinstance(urls, dict) or not urls:
        return jsonify({"error": "Request body must include a non-empty 'urls' object"}), 400
    logging.info("[google-eta] REQUEST received for routes: %s", ", ".join(sorted(map(str, urls.keys()))))

    etas_seconds = {}
    failures = {}
    details = {}
    sources = {}

    valid_items = []
    for name, url in urls.items():
        if not isinstance(name, str) or not isinstance(url, str) or not url.strip():
            failures[str(name)] = "Invalid URL payload"
            continue
        valid_items.append((name, url.strip()))

    with _google_scrape_lock:
        context = None
        page = None
        try:
            browser, _ = _ensure_google_browser()
            context = browser.new_context(viewport={"width": 1440, "height": 1200})
            context.route("**/*", _block_heavy_google_resources)
            page = context.new_page()
            page.set_default_timeout(10000)

            for name, url in valid_items:
                try:
                    scraped = _scrape_google_maps_eta_with_page(page, url, 10, name)
                    etas_seconds[name] = float(scraped["seconds"])
                    sources[name] = "google"
                    details[name] = {
                        "best_guess": scraped["best_guess"],
                        "candidate_count": scraped["candidate_count"],
                        "url": scraped["url"],
                    }
                except Exception as exc:
                    failures[name] = str(exc)
                    logging.warning("[google-eta] FAIL %s scrape: %s", name, exc)
        finally:
            try:
                if page is not None:
                    page.close()
            except Exception:
                pass
            try:
                if context is not None:
                    context.close()
            except Exception:
                pass

    logging.info(
        "[google-eta] RESPONSE etas=%s failures=%s sources=%s",
        {k: _format_duration_seconds(v) for k, v in etas_seconds.items()},
        failures,
        sources,
    )

    return jsonify({
        "etasSeconds": etas_seconds,
        "failures": failures,
        "details": details,
        "sources": sources,
    })


@app.route("/api/risk-definition", methods=["GET"])
def get_risk_definition():
    """
    Return percentile thresholds and short copy for what low/medium/high risk means.
    Lambda map is computed at startup; this endpoint only reads precomputed values.
    """
    if _lambda_p70 is None or _lambda_p90 is None:
        return jsonify({"error": "Risk thresholds not available"}), 503
    resp = {
        "p70": float(_lambda_p70),
        "p90": float(_lambda_p90),
        "description": (
            "Risk is based on predicted crash rate (λ) per segment. "
            "Low = bottom 70% of segments, Medium = 70th–90th percentile, High = top 10%."
        ),
        "low": "Bottom 70% of segments by predicted crash rate",
        "medium": "70th–90th percentile",
        "high": "Top 10% of segments by predicted crash rate",
    }
    importance = _get_feature_importance_map()
    if importance:
        resp["featureImportance"] = importance
    return jsonify(resp)


@app.route("/data-validation", methods=["GET"])
def data_validation_page():
    """Serve the data validation page"""
    return render_template_string(_get_data_validation_html())


def _convert_to_json_serializable(obj):
    """Convert numpy/pandas types to native Python types for JSON serialization"""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: _convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_convert_to_json_serializable(item) for item in obj]
    elif pd.isna(obj):
        return None
    else:
        return obj


@app.route("/api/fatality-diagnostic", methods=["GET"])
def fatality_diagnostic():
    """
    Diagnostic endpoint to check why no fatalities are being counted
    Examines the actual data structure and values
    """
    try:
        # Load raw data to inspect
        data_dir = PROJECT_ROOT / "data"

        diagnostic_info = {
            "error": None,
            "collision_data": {},
            "ksi_data": {},
            "sample_records": {},
        }

        try:
            # Actually load collision and KSI data
            from src.data_processing.data_loader import (
                load_collision_data,
                load_ksi_data,
            )

            try:
                collision_raw = load_collision_data(data_dir)
                diagnostic_info["collision_data"] = {
                    "total_records": len(collision_raw),
                    "columns": list(collision_raw.columns),
                    "fatalities_column": COLLISION_COLUMNS.get(
                        "fatalities", "FATALITIES"
                    ),
                    "injury_column": COLLISION_COLUMNS.get(
                        "injury", "INJURY_COLLISIONS"
                    ),
                    "has_fatalities_col": COLLISION_COLUMNS.get(
                        "fatalities", "FATALITIES"
                    )
                    in collision_raw.columns,
                    "has_injury_col": COLLISION_COLUMNS.get(
                        "injury", "INJURY_COLLISIONS"
                    )
                    in collision_raw.columns,
                }

                # Check what values exist in these columns
                fatalities_col = COLLISION_COLUMNS.get("fatalities", "FATALITIES")
                injury_col = COLLISION_COLUMNS.get("injury", "INJURY_COLLISIONS")

                if fatalities_col in collision_raw.columns:
                    non_zero_fatalities = (
                        collision_raw[collision_raw[fatalities_col] > 0]
                        if pd.api.types.is_numeric_dtype(collision_raw[fatalities_col])
                        else collision_raw[
                            collision_raw[fatalities_col].notna()
                            & (
                                collision_raw[fatalities_col].astype(str).str.lower()
                                != "0"
                            )
                        ]
                    )
                    diagnostic_info["collision_data"]["non_zero_fatalities_count"] = (
                        len(non_zero_fatalities)
                    )
                    fatalities_stats = {}
                    if len(collision_raw) > 0:
                        value_counts = (
                            collision_raw[fatalities_col]
                            .value_counts()
                            .head(10)
                            .to_dict()
                        )
                        fatalities_stats = {
                            "total_non_null": int(
                                collision_raw[fatalities_col].notna().sum()
                            ),
                            "unique_values": _convert_to_json_serializable(
                                value_counts
                            ),
                            "sample_values": _convert_to_json_serializable(
                                collision_raw[fatalities_col].head(10).tolist()
                            ),
                        }
                    else:
                        fatalities_stats = {
                            "total_non_null": 0,
                            "unique_values": {},
                            "sample_values": [],
                        }
                    diagnostic_info["collision_data"][
                        "fatalities_column_stats"
                    ] = fatalities_stats

                if injury_col in collision_raw.columns:
                    fatal_in_injury = collision_raw[
                        collision_raw[injury_col]
                        .astype(str)
                        .str.contains("Fatal", case=False, na=False)
                    ]
                    diagnostic_info["collision_data"]["fatal_in_injury_count"] = len(
                        fatal_in_injury
                    )
                    if len(collision_raw) > 0:
                        injury_values = (
                            collision_raw[injury_col].value_counts().head(10).to_dict()
                        )
                        diagnostic_info["collision_data"][
                            "injury_column_unique_values"
                        ] = _convert_to_json_serializable(injury_values)
                    else:
                        diagnostic_info["collision_data"][
                            "injury_column_unique_values"
                        ] = {}

                # Sample records with potential fatalities
                if fatalities_col in collision_raw.columns:
                    sample_fatal = (
                        collision_raw.nlargest(5, fatalities_col)
                        if pd.api.types.is_numeric_dtype(collision_raw[fatalities_col])
                        else collision_raw.head(5)
                    )
                    diagnostic_info["sample_records"]["collision_with_fatalities"] = [
                        {
                            "fatalities_col_value": str(
                                record.get(fatalities_col, "N/A")
                            ),
                            "injury_col_value": (
                                str(record.get(injury_col, "N/A"))
                                if injury_col in record.index
                                else "N/A"
                            ),
                            "columns_with_fatal": [
                                col
                                for col in record.index
                                if "fatal" in col.lower() or "death" in col.lower()
                            ],
                        }
                        for idx, record in sample_fatal.iterrows()
                    ]

            except Exception as e:
                diagnostic_info["collision_data"]["error"] = str(e)

            try:
                ksi_raw = load_ksi_data(data_dir)
                diagnostic_info["ksi_data"] = {
                    "total_records": len(ksi_raw),
                    "columns": list(ksi_raw.columns),
                    "fatalities_column": KSI_COLUMNS.get("fatalities", "FATAL_NO"),
                    "injury_column": KSI_COLUMNS.get("injury", "INJURY"),
                    "has_fatalities_col": KSI_COLUMNS.get("fatalities", "FATAL_NO")
                    in ksi_raw.columns,
                    "has_injury_col": KSI_COLUMNS.get("injury", "INJURY")
                    in ksi_raw.columns,
                }

                # Check what values exist in these columns
                fatalities_col = KSI_COLUMNS.get("fatalities", "FATAL_NO")
                injury_col = KSI_COLUMNS.get("injury", "INJURY")

                if fatalities_col in ksi_raw.columns:
                    non_zero_fatalities = (
                        ksi_raw[ksi_raw[fatalities_col] > 0]
                        if pd.api.types.is_numeric_dtype(ksi_raw[fatalities_col])
                        else ksi_raw[
                            ksi_raw[fatalities_col].notna()
                            & (ksi_raw[fatalities_col].astype(str).str.lower() != "0")
                        ]
                    )
                    diagnostic_info["ksi_data"]["non_zero_fatalities_count"] = len(
                        non_zero_fatalities
                    )
                    ksi_fatalities_stats = {}
                    if len(ksi_raw) > 0:
                        value_counts = (
                            ksi_raw[fatalities_col].value_counts().head(10).to_dict()
                        )
                        ksi_fatalities_stats = {
                            "total_non_null": int(
                                ksi_raw[fatalities_col].notna().sum()
                            ),
                            "unique_values": _convert_to_json_serializable(
                                value_counts
                            ),
                            "sample_values": _convert_to_json_serializable(
                                ksi_raw[fatalities_col].head(10).tolist()
                            ),
                            "data_type": str(ksi_raw[fatalities_col].dtype),
                        }
                    else:
                        ksi_fatalities_stats = {
                            "total_non_null": 0,
                            "unique_values": {},
                            "sample_values": [],
                            "data_type": "unknown",
                        }
                    diagnostic_info["ksi_data"][
                        "fatalities_column_stats"
                    ] = ksi_fatalities_stats

                if injury_col in ksi_raw.columns:
                    fatal_in_injury = ksi_raw[
                        ksi_raw[injury_col]
                        .astype(str)
                        .str.contains("Fatal", case=False, na=False)
                    ]
                    diagnostic_info["ksi_data"]["fatal_in_injury_count"] = len(
                        fatal_in_injury
                    )
                    if len(ksi_raw) > 0:
                        injury_values = (
                            ksi_raw[injury_col].value_counts().head(10).to_dict()
                        )
                        diagnostic_info["ksi_data"]["injury_column_unique_values"] = (
                            _convert_to_json_serializable(injury_values)
                        )
                    else:
                        diagnostic_info["ksi_data"]["injury_column_unique_values"] = {}

                # Sample records
                if fatalities_col in ksi_raw.columns:
                    sample_fatal = (
                        ksi_raw.nlargest(5, fatalities_col)
                        if pd.api.types.is_numeric_dtype(ksi_raw[fatalities_col])
                        else ksi_raw.head(5)
                    )
                    diagnostic_info["sample_records"]["ksi_with_fatalities"] = [
                        {
                            "fatalities_col_value": str(
                                record.get(fatalities_col, "N/A")
                            ),
                            "injury_col_value": (
                                str(record.get(injury_col, "N/A"))
                                if injury_col in record.index
                                else "N/A"
                            ),
                            "columns_with_fatal": [
                                col
                                for col in record.index
                                if "fatal" in col.lower() or "death" in col.lower()
                            ],
                        }
                        for idx, record in sample_fatal.iterrows()
                    ]

            except Exception as e:
                diagnostic_info["ksi_data"]["error"] = str(e)

        except Exception as e:
            diagnostic_info["error"] = str(e)
            import traceback

            diagnostic_info["traceback"] = traceback.format_exc()

        return jsonify(diagnostic_info)

    except Exception as e:
        logging.error(f"Error in fatality diagnostic: {e}")
        import traceback

        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route("/api/model-features", methods=["GET"])
def get_model_features():
    """
    Get the list of features/inputs used by the temporal count model
    """
    if not _is_temporal_model_loaded():
        return jsonify({"error": "Temporal model not loaded"}), 500

    try:
        feature_columns = getattr(temporal_trainer, "feature_columns", None) or []

        temporal_features = [
            f
            for f in feature_columns
            if any(
                x in f.lower()
                for x in ["time", "season", "weekend", "hour", "month", "day"]
            )
        ]
        road_features = [
            f
            for f in feature_columns
            if any(x in f.lower() for x in ["road", "class", "length", "segment"])
        ]
        other_features = [
            f
            for f in feature_columns
            if f not in temporal_features and f not in road_features
        ]

        feature_importance = {}
        # Hurdle has stage2; single-stage has model
        model = getattr(temporal_trainer, "model", None) or getattr(
            temporal_trainer, "stage2", None
        )
        if (
            model is not None
            and hasattr(model, "feature_importances_")
            and feature_columns
        ):
            importances = model.feature_importances_
            feature_importance = dict(zip(feature_columns, importances.tolist()))
            feature_importance = dict(
                sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            )

        return jsonify(
            {
                "total_features": len(feature_columns),
                "feature_columns": feature_columns,
                "feature_categories": {
                    "temporal_features": temporal_features,
                    "road_characteristics": road_features,
                    "other_features": other_features,
                },
                "feature_importance": feature_importance,
                "model_type": "temporal_count",
                "note": "Temporal count model predicts crash rate λ per segment-window. Features come from panel (road, temporal, weather, lag).",
            }
        )
    except Exception as e:
        logging.error(f"Error getting model features: {e}")
        import traceback

        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/data-verification", methods=["GET"])
def data_verification():
    """
    Provide diagnostic information about road network and segment counts.
    Crash/KSI/fatality stats require the spatial-join pipeline; raw road network has none.
    """
    if road_network is None:
        return jsonify({"error": "No road data available"}), 500

    try:
        data_source = road_network

        # Analyze the data relationships
        total_segments = len(data_source)
        segments_with_crashes = (
            len(data_source[data_source.get("num_total_crashes", 0) > 0])
            if "num_total_crashes" in data_source.columns
            else 0
        )
        segments_with_ksi = (
            len(data_source[data_source.get("num_ksi_crashes", 0) > 0])
            if "num_ksi_crashes" in data_source.columns
            else 0
        )
        segments_with_fatalities = (
            len(data_source[data_source.get("fatality_count", 0) > 0])
            if "fatality_count" in data_source.columns
            else 0
        )

        # Calculate totals
        total_crashes = (
            int(data_source.get("num_total_crashes", 0).sum())
            if "num_total_crashes" in data_source.columns
            else 0
        )
        total_ksi = (
            int(data_source.get("num_ksi_crashes", 0).sum())
            if "num_ksi_crashes" in data_source.columns
            else 0
        )
        total_fatalities = (
            int(data_source.get("fatality_count", 0).sum())
            if "fatality_count" in data_source.columns
            else 0
        )

        # Check if fatality counts are likely accurate
        # Based on the diagnostic, we expect:
        # - Collision data: 591 records with non-zero fatalities
        # - KSI data: 870 records with non-zero fatalities
        # Total should be in the hundreds, not zero
        fatality_accuracy = {
            "total_fatalities_in_data": total_fatalities,
            "segments_with_fatalities": segments_with_fatalities,
            "expected_range": "Hundreds to thousands (based on 591 + 870 raw records)",
            "likely_accurate": total_fatalities > 100,  # Reasonable threshold
            "warning": "If total_fatalities is 0 or very low (<50), the data needs to be regenerated with the fixed fatality counting logic.",
        }

        # Find segments where KSI > Total crashes (data inconsistency)
        inconsistencies = []
        if (
            "num_total_crashes" in data_source.columns
            and "num_ksi_crashes" in data_source.columns
        ):
            inconsistent = data_source[
                data_source["num_ksi_crashes"] > data_source["num_total_crashes"]
            ]
            inconsistencies = [
                {
                    "id": str(seg.get("segment_id", idx)),
                    "street": str(seg.get("LINEAR_NAME", "Unknown")),
                    "total_crashes": int(seg.get("num_total_crashes", 0)),
                    "ksi_crashes": int(seg.get("num_ksi_crashes", 0)),
                }
                for idx, seg in inconsistent.head(10).iterrows()
            ]

        # Sample segments to analyze
        sample_segments = []
        if "num_total_crashes" in data_source.columns:
            # Get segments with high crash counts but low/no KSI
            high_crash_low_ksi = data_source[
                (data_source["num_total_crashes"] >= 20)
                & (data_source.get("num_ksi_crashes", 0) == 0)
            ].head(5)

            sample_segments = [
                {
                    "id": str(seg.get("segment_id", idx)),
                    "street": str(seg.get("LINEAR_NAME", "Unknown")),
                    "total_crashes": int(seg.get("num_total_crashes", 0)),
                    "ksi_crashes": (
                        int(seg.get("num_ksi_crashes", 0))
                        if "num_ksi_crashes" in data_source.columns
                        else 0
                    ),
                    "fatalities": (
                        int(seg.get("fatality_count", 0))
                        if "fatality_count" in data_source.columns
                        else 0
                    ),
                }
                for idx, seg in high_crash_low_ksi.iterrows()
            ]

        return jsonify(
            {
                "summary": {
                    "total_segments": total_segments,
                    "segments_with_crashes": segments_with_crashes,
                    "segments_with_ksi": segments_with_ksi,
                    "total_crashes": total_crashes,
                    "total_ksi_crashes": total_ksi,
                    "total_fatalities": total_fatalities,
                    "segments_with_fatalities": segments_with_fatalities,
                    "ksi_ratio": (
                        round(total_ksi / total_crashes * 100, 2)
                        if total_crashes > 0
                        else 0
                    ),
                },
                "fatality_accuracy_check": fatality_accuracy,
                "data_structure_explanation": {
                    "total_crashes_source": "Count of records from collision dataset (all traffic collisions)",
                    "ksi_crashes_source": "Count of records from KSI dataset (crashes with killed/seriously injured persons)",
                    "note": "KSI and collision datasets are counted separately. KSI crashes may overlap with total crashes if they represent the same incidents, or may be separate records. In typical traffic data, KSI crashes are a subset of total crashes.",
                },
                "inconsistencies": {
                    "count": len(inconsistencies),
                    "examples": inconsistencies,
                    "note": "These segments have more KSI crashes than total crashes, which may indicate data issues.",
                },
                "sample_high_crash_low_ksi": {
                    "count": len(sample_segments),
                    "examples": sample_segments,
                    "note": "Sample segments with high crash counts but no KSI crashes.",
                },
            }
        )
    except Exception as e:
        logging.error(f"Error in data verification: {e}")
        import traceback

        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/street-names", methods=["GET"])
def get_street_names():
    """
    Get street name suggestions for autocomplete

    Query parameters:
    - query: search query (required, at least 2 characters)
    - limit: maximum number of results (default: 20, max: 100)
    """
    query = request.args.get("query", "").strip()

    if len(query) < 2:
        return jsonify({"suggestions": []})

    limit = min(int(request.args.get("limit", 20)), 100)

    data_source = road_network

    if data_source is None:
        return jsonify({"suggestions": []})

    try:
        # Get unique street names matching the query
        if "LINEAR_NAME" in data_source.columns:
            matching_names = data_source[
                data_source["LINEAR_NAME"].str.contains(query, case=False, na=False)
            ]["LINEAR_NAME"].unique()

            # Sort by length (shorter/more exact matches first) then alphabetically
            matching_names = sorted(matching_names, key=lambda x: (len(x), x.lower()))[
                :limit
            ]

            suggestions = [{"name": name, "value": name} for name in matching_names]
            return jsonify({"suggestions": suggestions})
        else:
            return jsonify({"suggestions": []})

    except Exception as e:
        logging.error(f"Error in get_street_names: {e}")
        return jsonify({"suggestions": []})


@app.route("/api/segments/all", methods=["GET"])
def get_all_segments():
    """
    Get all segments with optional filtering and pagination

    Query parameters:
    - page: page number (default: 1)
    - per_page: items per page (default: 100, max: 1000)
    - search: search in LINEAR_NAME (optional)
    - risk_label: filter by risk label (low/medium/high, optional)
    - min_crashes: minimum total crashes (optional)
    - sort_by: column to sort by (default: id)
    - sort_order: asc or desc (default: asc)
    """
    data_source = road_network

    if data_source is None:
        return (
            jsonify(
                {
                    "error": "No road data available. Please ensure road network data is loaded."
                }
            ),
            500,
        )

    try:
        # Get query parameters
        page = int(request.args.get("page", 1))
        per_page = min(int(request.args.get("per_page", 100)), 1000)
        search = request.args.get("search", "").strip()
        risk_label = request.args.get("risk_label", "").strip().lower()
        min_crashes = request.args.get("min_crashes")
        sort_by = request.args.get("sort_by", "segment_id")
        sort_order = request.args.get("sort_order", "asc").lower()

        # Start with all data
        filtered_data = data_source.copy()

        # Apply search filter
        if search:
            filtered_data = filtered_data[
                filtered_data["LINEAR_NAME"].str.contains(search, case=False, na=False)
            ]

        # Apply risk label filter (only if risk_label column exists)
        if risk_label and risk_label in ["low", "medium", "high"]:
            if "risk_label" in filtered_data.columns:
                filtered_data = filtered_data[filtered_data["risk_label"] == risk_label]

        # Apply minimum crashes filter (only if column exists)
        if min_crashes:
            try:
                min_crashes = int(min_crashes)
                if "num_total_crashes" in filtered_data.columns:
                    filtered_data = filtered_data[
                        filtered_data["num_total_crashes"] >= min_crashes
                    ]
            except ValueError:
                pass

        # Prepare data for sorting
        ascending = sort_order == "asc"

        # Validate sort column exists
        if sort_by not in filtered_data.columns:
            # If trying to sort by segment_id but it doesn't exist, try to use index
            if sort_by == "segment_id":
                # Sort by index instead
                filtered_data = filtered_data.sort_index(ascending=ascending)
            else:
                # Default to first numeric column or index
                numeric_cols = filtered_data.select_dtypes(
                    include=[np.number]
                ).columns.tolist()
                if numeric_cols:
                    sort_by = numeric_cols[0]
                else:
                    # Fallback to index-based sorting
                    filtered_data = filtered_data.sort_index(ascending=ascending)
                    sort_by = None  # Mark as already sorted

        # Sort data if not already sorted by index
        if sort_by:
            try:
                filtered_data = filtered_data.sort_values(
                    by=sort_by, ascending=ascending, na_position="last"
                )
            except (KeyError, ValueError) as e:
                # Fallback to index sorting if column sorting fails
                logging.warning(f"Could not sort by {sort_by}, using index: {e}")
                filtered_data = filtered_data.sort_index(ascending=ascending)

        # Calculate pagination
        total = len(filtered_data)
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        paginated_data = filtered_data.iloc[start_idx:end_idx]

        # Convert to JSON-serializable format
        results = []
        for idx, segment in paginated_data.iterrows():
            coords = _extract_coordinates(segment.geometry)

            # Get all relevant columns (handle missing columns gracefully)
            # Calculate segment length from geometry if not available
            segment_length = segment.get("segment_length", 0)
            if pd.isna(segment_length) or segment_length == 0:
                try:
                    if hasattr(segment.geometry, "length"):
                        segment_length = (
                            segment.geometry.length * 111000
                        )  # Approximate meters (rough conversion)
                except:
                    segment_length = 0

            confidence = segment.get("confidence", None)
            risk_label = "low"
            if (
                _is_temporal_model_loaded()
                and panel_data is not None
                and not panel_data.empty
            ):
                try:
                    if lambda_per_hour_latest is None:
                        _compute_lambda_map_for_latest_window()
                    seg_id = segment.get("segment_id") or segment.get(
                        "CENTRELINE_ID", idx
                    )
                    lam = lambda_per_hour_latest.get(seg_id, 0.0)  # type: ignore[union-attr]
                    if (
                        lam == 0.0
                        and isinstance(seg_id, float)
                        and not np.isnan(seg_id)
                    ):
                        lam = lambda_per_hour_latest.get(int(seg_id), 0.0)  # type: ignore[union-attr]
                    risk_label = _lambda_to_risk_label(lam)
                    wh = temporal_trainer.panel_config.window_size_hours or 1  # type: ignore[union-attr]
                    prob_crash = 1.0 - np.exp(-max(0, lam * wh))
                    confidence = float(np.clip(prob_crash, 0.0, 1.0))
                except Exception as e:
                    logging.debug(
                        f"Could not compute temporal risk for segment {idx}: {e}"
                    )
                    confidence = 0.5
            else:
                confidence = (
                    float(confidence)
                    if confidence is not None and pd.notna(confidence)
                    else 0.5
                )

            segment_dict = {
                "id": str(segment.get("segment_id", idx)),
                "LINEAR_NAME": str(segment.get("LINEAR_NAME", "Unknown")),
                "ROAD_CLASS": str(
                    segment.get(
                        "ROAD_CLASS", segment.get("LINEAR_NAME_TYPE", "Unknown")
                    )
                ),
                "segment_length": float(segment_length),
                "risk_label": risk_label,
                "confidence": confidence,
                "num_total_crashes": (
                    int(segment.get("num_total_crashes", 0))
                    if "num_total_crashes" in segment
                    else 0
                ),
                "num_ksi_crashes": (
                    int(segment.get("num_ksi_crashes", 0))
                    if "num_ksi_crashes" in segment
                    else 0
                ),
                "fatality_count": (
                    int(segment.get("fatality_count", 0))
                    if "fatality_count" in segment
                    else 0
                ),
                "ksi_ratio": (
                    float(segment.get("ksi_ratio", 0))
                    if "ksi_ratio" in segment
                    else 0.0
                ),
                "crash_density": (
                    float(segment.get("crash_density", 0))
                    if "crash_density" in segment
                    else 0.0
                ),
                "coordinates": coords,
            }
            results.append(segment_dict)

        return jsonify(
            {
                "data": results,
                "pagination": {
                    "page": page,
                    "per_page": per_page,
                    "total": total,
                    "total_pages": (total + per_page - 1) // per_page,
                },
            }
        )

    except Exception as e:
        logging.error(f"Error in get_all_segments: {e}")
        import traceback

        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


def _get_data_validation_html():
    """Return HTML template for data validation page"""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Street Segment Data Validation</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: #f5f5f7;
            color: #1d1d1f;
            padding: 20px;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            padding: 30px;
        }
        
        h1 {
            font-size: 32px;
            margin-bottom: 10px;
            color: #1d1d1f;
        }
        
        .subtitle {
            color: #86868b;
            margin-bottom: 30px;
            font-size: 14px;
        }
        
        .controls {
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
            margin-bottom: 25px;
            padding: 20px;
            background: #f5f5f7;
            border-radius: 8px;
        }
        
        .control-group {
            display: flex;
            flex-direction: column;
            gap: 5px;
        }
        
        .control-group label {
            font-size: 12px;
            font-weight: 600;
            color: #6e6e73;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        input, select {
            padding: 8px 12px;
            border: 1px solid #d2d2d7;
            border-radius: 6px;
            font-size: 14px;
            transition: all 0.2s;
        }
        
        input:focus, select:focus {
            outline: none;
            border-color: #0071e3;
            box-shadow: 0 0 0 3px rgba(0,113,227,0.1);
        }
        
        .search-input {
            min-width: 250px;
        }
        
        .autocomplete-container {
            position: relative;
            width: 100%;
        }
        
        .autocomplete-dropdown {
            position: absolute;
            top: 100%;
            left: 0;
            right: 0;
            background: white;
            border: 1px solid #d2d2d7;
            border-top: none;
            border-radius: 0 0 6px 6px;
            max-height: 200px;
            overflow-y: auto;
            z-index: 1000;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            display: none;
        }
        
        .autocomplete-dropdown.show {
            display: block;
        }
        
        .autocomplete-item {
            padding: 10px 12px;
            cursor: pointer;
            border-bottom: 1px solid #f5f5f7;
            transition: background 0.15s;
        }
        
        .autocomplete-item:hover,
        .autocomplete-item.selected {
            background: #f5f5f7;
        }
        
        .autocomplete-item:last-child {
            border-bottom: none;
        }
        
        .autocomplete-item strong {
            color: #0071e3;
        }
        
        .stats {
            display: flex;
            gap: 20px;
            margin-bottom: 25px;
            flex-wrap: wrap;
        }
        
        .stat-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            min-width: 150px;
        }
        
        .stat-card.high-risk {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        }
        
        .stat-card.medium-risk {
            background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        }
        
        .stat-card.low-risk {
            background: linear-gradient(135deg, #30cfd0 0%, #330867 100%);
        }
        
        .stat-label {
            font-size: 12px;
            opacity: 0.9;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .stat-value {
            font-size: 28px;
            font-weight: 600;
            margin-top: 5px;
        }
        
        .table-container {
            overflow-x: auto;
            border-radius: 8px;
            border: 1px solid #d2d2d7;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }
        
        thead {
            background: #f5f5f7;
            position: sticky;
            top: 0;
        }
        
        th {
            padding: 12px 15px;
            text-align: left;
            font-weight: 600;
            color: #6e6e73;
            text-transform: uppercase;
            font-size: 11px;
            letter-spacing: 0.5px;
            border-bottom: 2px solid #d2d2d7;
            cursor: pointer;
            user-select: none;
        }
        
        th:hover {
            background: #e8e8ed;
        }
        
        th.sorted {
            color: #0071e3;
        }
        
        td {
            padding: 12px 15px;
            border-bottom: 1px solid #e8e8ed;
        }
        
        tbody tr:hover {
            background: #f9f9fb;
        }
        
        tbody tr:last-child td {
            border-bottom: none;
        }
        
        .risk-badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
        }
        
        .risk-high {
            background: #ff3b30;
            color: white;
        }
        
        .risk-medium {
            background: #ff9500;
            color: white;
        }
        
        .risk-low {
            background: #34c759;
            color: white;
        }
        
        .pagination {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-top: 20px;
            padding: 15px;
            background: #f5f5f7;
            border-radius: 8px;
        }
        
        .pagination-info {
            color: #6e6e73;
            font-size: 14px;
        }
        
        .pagination-controls {
            display: flex;
            gap: 10px;
        }
        
        button {
            padding: 8px 16px;
            background: #0071e3;
            color: white;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 500;
            transition: all 0.2s;
        }
        
        button:hover:not(:disabled) {
            background: #0077ed;
            transform: translateY(-1px);
            box-shadow: 0 2px 8px rgba(0,113,227,0.3);
        }
        
        button:disabled {
            background: #d2d2d7;
            cursor: not-allowed;
            opacity: 0.6;
        }
        
        .loading {
            text-align: center;
            padding: 40px;
            color: #6e6e73;
        }
        
        .error {
            background: #ff3b30;
            color: white;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
        }
        
        .num-cell {
            text-align: right;
            font-variant-numeric: tabular-nums;
        }
        
        .coordinates-info {
            font-size: 11px;
            color: #6e6e73;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Street Segment Data Validation</h1>
        <p class="subtitle">Review all street segments and their associated risk data for accuracy</p>
        
        <div class="controls">
            <div class="control-group">
                <label>Search Street Name</label>
                <div class="autocomplete-container">
                    <input type="text" id="searchInput" class="search-input" placeholder="Enter street name..." autocomplete="off">
                    <div class="autocomplete-dropdown" id="autocompleteDropdown"></div>
                </div>
            </div>
            
            <div class="control-group">
                <label>Risk Level</label>
                <select id="riskFilter">
                    <option value="">All Risk Levels</option>
                    <option value="high">High Risk</option>
                    <option value="medium">Medium Risk</option>
                    <option value="low">Low Risk</option>
                </select>
            </div>
            
            <div class="control-group">
                <label>Min Crashes</label>
                <input type="number" id="minCrashes" placeholder="0" min="0">
            </div>
            
            <div class="control-group">
                <label>Sort By</label>
                <select id="sortBy">
                    <option value="segment_id">ID</option>
                    <option value="LINEAR_NAME">Street Name</option>
                    <option value="risk_label">Risk Level</option>
                    <option value="num_total_crashes">Total Crashes</option>
                    <option value="confidence">Confidence</option>
                    <option value="segment_length">Length</option>
                </select>
            </div>
            
            <div class="control-group">
                <label>Order</label>
                <select id="sortOrder">
                    <option value="asc">Ascending</option>
                    <option value="desc">Descending</option>
                </select>
            </div>
            
            <div class="control-group">
                <label>Per Page</label>
                <select id="perPage">
                    <option value="50">50</option>
                    <option value="100" selected>100</option>
                    <option value="250">250</option>
                    <option value="500">500</option>
                    <option value="1000">1000</option>
                </select>
            </div>
        </div>
        
        <div class="stats" id="stats">
            <!-- Stats will be populated dynamically -->
        </div>
        
        <div style="background: #fff3cd; border: 1px solid #ffc107; border-radius: 8px; padding: 15px; margin-bottom: 20px;">
            <h3 style="margin-top: 0; color: #856404;">📊 Data Verification Note</h3>
            <p style="margin-bottom: 10px; color: #856404;">
                <strong>Understanding the crash data structure:</strong>
            </p>
            <ul style="margin-bottom: 10px; color: #856404; padding-left: 20px;">
                <li><strong>Total Crashes:</strong> Count of all records from the collision dataset</li>
                <li><strong>KSI Crashes:</strong> Count of all records from the KSI (Killed/Seriously Injured) dataset</li>
                <li><strong>Fatalities:</strong> Count of fatalities from both datasets</li>
            </ul>
            <p style="margin-bottom: 10px; color: #856404;">
                <strong>Important:</strong> The collision and KSI datasets are counted separately. KSI crashes may:
            </p>
            <ul style="margin-bottom: 10px; color: #856404; padding-left: 20px;">
                <li>Be a subset of total crashes (if KSI records are also in the collision dataset)</li>
                <li>Be separate records (if KSI dataset contains different crashes)</li>
            </ul>
            <p style="margin-bottom: 0;">
                <button onclick="showDataVerification()" style="background: #856404; color: white; border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer; margin-right: 10px;">
                    View Data Verification Details
                </button>
                <button onclick="showFatalityDiagnostic()" style="background: #dc3545; color: white; border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer; margin-right: 10px;">
                    🔍 Diagnose Fatality Data Issue
                </button>
                <button onclick="showModelFeatures()" style="background: #6c757d; color: white; border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer;">
                    📊 View Model Features
                </button>
            </p>
            <div id="verificationDetails" style="display: none; margin-top: 15px; padding: 10px; background: white; border-radius: 4px;"></div>
            <div id="fatalityDiagnosticDetails" style="display: none; margin-top: 15px; padding: 10px; background: white; border-radius: 4px;"></div>
            <div id="modelFeaturesDetails" style="display: none; margin-top: 15px; padding: 10px; background: white; border-radius: 4px;"></div>
        </div>
        
        <div id="errorContainer"></div>
        <div id="loadingMessage" class="loading">Loading data...</div>
        
        <div class="table-container" id="tableContainer" style="display: none;">
            <table>
                <thead>
                    <tr>
                        <th data-sort="segment_id">ID</th>
                        <th data-sort="LINEAR_NAME">Street Name</th>
                        <th data-sort="ROAD_CLASS">Road Class</th>
                        <th data-sort="segment_length">Length (m)</th>
                        <th data-sort="risk_label">Risk</th>
                        <th data-sort="confidence">Confidence</th>
                        <th data-sort="num_total_crashes">Total Crashes</th>
                        <th data-sort="num_ksi_crashes">KSI Crashes</th>
                        <th data-sort="fatality_count">Fatalities</th>
                        <th data-sort="crash_density">Crash Density</th>
                        <th>Coordinates</th>
                    </tr>
                </thead>
                <tbody id="tableBody">
                    <!-- Table rows will be populated dynamically -->
                </tbody>
            </table>
        </div>
        
        <div class="pagination" id="pagination" style="display: none;">
            <div class="pagination-info" id="paginationInfo"></div>
            <div class="pagination-controls">
                <button id="prevPage" disabled>Previous</button>
                <button id="nextPage" disabled>Next</button>
            </div>
        </div>
    </div>
    
    <script>
        let currentPage = 1;
        let currentFilters = {};
        let autocompleteTimeout = null;
        let selectedAutocompleteIndex = -1;
        
        // Initialize
        document.addEventListener('DOMContentLoaded', () => {
            setupEventListeners();
            loadData();
        });
        
        function setupEventListeners() {
            const searchInput = document.getElementById('searchInput');
            
            // Autocomplete on input
            searchInput.addEventListener('input', (e) => {
                const query = e.target.value.trim();
                if (query.length >= 2) {
                    clearTimeout(autocompleteTimeout);
                    autocompleteTimeout = setTimeout(() => {
                        fetchAutocompleteSuggestions(query);
                    }, 300);
                } else {
                    hideAutocomplete();
                }
                handleFilterChange();
            });
            
            // Keyboard navigation for autocomplete
            searchInput.addEventListener('keydown', (e) => {
                const dropdown = document.getElementById('autocompleteDropdown');
                const items = dropdown.querySelectorAll('.autocomplete-item');
                
                if (e.key === 'ArrowDown') {
                    e.preventDefault();
                    selectedAutocompleteIndex = Math.min(selectedAutocompleteIndex + 1, items.length - 1);
                    updateAutocompleteSelection(items);
                } else if (e.key === 'ArrowUp') {
                    e.preventDefault();
                    selectedAutocompleteIndex = Math.max(selectedAutocompleteIndex - 1, -1);
                    updateAutocompleteSelection(items);
                } else if (e.key === 'Enter' && selectedAutocompleteIndex >= 0) {
                    e.preventDefault();
                    if (items[selectedAutocompleteIndex]) {
                        items[selectedAutocompleteIndex].click();
                    }
                } else if (e.key === 'Escape') {
                    hideAutocomplete();
                }
            });
            
            // Hide autocomplete when clicking outside
            document.addEventListener('click', (e) => {
                if (!e.target.closest('.autocomplete-container')) {
                    hideAutocomplete();
                }
            });
            
            document.getElementById('riskFilter').addEventListener('change', handleFilterChange);
            document.getElementById('minCrashes').addEventListener('input', debounce(handleFilterChange, 500));
            document.getElementById('sortBy').addEventListener('change', handleFilterChange);
            document.getElementById('sortOrder').addEventListener('change', handleFilterChange);
            document.getElementById('perPage').addEventListener('change', () => {
                currentPage = 1;
                handleFilterChange();
            });
            
            document.getElementById('prevPage').addEventListener('click', () => {
                if (currentPage > 1) {
                    currentPage--;
                    loadData();
                }
            });
            
            document.getElementById('nextPage').addEventListener('click', () => {
                currentPage++;
                loadData();
            });
            
            // Table header sorting
            document.querySelectorAll('th[data-sort]').forEach(th => {
                th.addEventListener('click', () => {
                    const sortBy = th.getAttribute('data-sort');
                    const currentSortBy = document.getElementById('sortBy').value;
                    const currentOrder = document.getElementById('sortOrder').value;
                    
                    if (currentSortBy === sortBy) {
                        document.getElementById('sortOrder').value = currentOrder === 'asc' ? 'desc' : 'asc';
                    } else {
                        document.getElementById('sortBy').value = sortBy;
                        document.getElementById('sortOrder').value = 'asc';
                    }
                    
                    handleFilterChange();
                });
            });
        }
        
        function handleFilterChange() {
            currentPage = 1;
            loadData();
        }
        
        async function fetchAutocompleteSuggestions(query) {
            try {
                const response = await fetch(`/api/street-names?query=${encodeURIComponent(query)}&limit=20`);
                const result = await response.json();
                displayAutocompleteSuggestions(result.suggestions || [], query);
            } catch (error) {
                console.error('Error fetching autocomplete suggestions:', error);
                hideAutocomplete();
            }
        }
        
        function displayAutocompleteSuggestions(suggestions, query) {
            const dropdown = document.getElementById('autocompleteDropdown');
            dropdown.innerHTML = '';
            selectedAutocompleteIndex = -1;
            
            if (suggestions.length === 0) {
                hideAutocomplete();
                return;
            }
            
            suggestions.forEach((suggestion, index) => {
                const item = document.createElement('div');
                item.className = 'autocomplete-item';
                
                // Highlight matching part
                const name = suggestion.name;
                const queryLower = query.toLowerCase();
                const nameLower = name.toLowerCase();
                const matchIndex = nameLower.indexOf(queryLower);
                
                if (matchIndex >= 0) {
                    const beforeMatch = name.substring(0, matchIndex);
                    const match = name.substring(matchIndex, matchIndex + query.length);
                    const afterMatch = name.substring(matchIndex + query.length);
                    item.innerHTML = `${beforeMatch}<strong>${match}</strong>${afterMatch}`;
                } else {
                    item.textContent = name;
                }
                
                item.addEventListener('click', () => {
                    document.getElementById('searchInput').value = name;
                    hideAutocomplete();
                    handleFilterChange();
                });
                
                dropdown.appendChild(item);
            });
            
            dropdown.classList.add('show');
        }
        
        function updateAutocompleteSelection(items) {
            items.forEach((item, index) => {
                if (index === selectedAutocompleteIndex) {
                    item.classList.add('selected');
                    item.scrollIntoView({ block: 'nearest' });
                } else {
                    item.classList.remove('selected');
                }
            });
        }
        
        function hideAutocomplete() {
            const dropdown = document.getElementById('autocompleteDropdown');
            dropdown.classList.remove('show');
            selectedAutocompleteIndex = -1;
        }
        
        function debounce(func, wait) {
            let timeout;
            return function executedFunction(...args) {
                const later = () => {
                    clearTimeout(timeout);
                    func(...args);
                };
                clearTimeout(timeout);
                timeout = setTimeout(later, wait);
            };
        }
        
        async function loadData() {
            const search = document.getElementById('searchInput').value;
            const riskLabel = document.getElementById('riskFilter').value;
            const minCrashes = document.getElementById('minCrashes').value;
            const sortBy = document.getElementById('sortBy').value;
            const sortOrder = document.getElementById('sortOrder').value;
            const perPage = document.getElementById('perPage').value;
            
            const params = new URLSearchParams({
                page: currentPage,
                per_page: perPage,
                sort_by: sortBy,
                sort_order: sortOrder
            });
            
            if (search) params.append('search', search);
            if (riskLabel) params.append('risk_label', riskLabel);
            if (minCrashes) params.append('min_crashes', minCrashes);
            
            document.getElementById('loadingMessage').style.display = 'block';
            document.getElementById('tableContainer').style.display = 'none';
            document.getElementById('pagination').style.display = 'none';
            document.getElementById('errorContainer').innerHTML = '';
            
            try {
                const response = await fetch(`/api/segments/all?${params}`);
                const result = await response.json();
                
                if (!response.ok) {
                    throw new Error(result.error || 'Failed to load data');
                }
                
                displayData(result);
                updateStats(result);
                updatePagination(result.pagination);
                
            } catch (error) {
                document.getElementById('errorContainer').innerHTML = 
                    `<div class="error">Error: ${error.message}</div>`;
            } finally {
                document.getElementById('loadingMessage').style.display = 'none';
            }
        }
        
        function displayData(result) {
            const tbody = document.getElementById('tableBody');
            tbody.innerHTML = '';
            
            if (result.data.length === 0) {
                tbody.innerHTML = '<tr><td colspan="11" style="text-align: center; padding: 40px; color: #6e6e73;">No segments found matching your filters</td></tr>';
                document.getElementById('tableContainer').style.display = 'block';
                return;
            }
            
            result.data.forEach(segment => {
                const row = document.createElement('tr');
                row.innerHTML = `
                    <td>${segment.id}</td>
                    <td>${segment.LINEAR_NAME}</td>
                    <td>${segment.ROAD_CLASS}</td>
                    <td class="num-cell">${segment.segment_length.toFixed(2)}</td>
                    <td><span class="risk-badge risk-${segment.risk_label}">${segment.risk_label}</span></td>
                    <td class="num-cell">${(segment.confidence * 100).toFixed(1)}%</td>
                    <td class="num-cell">${segment.num_total_crashes}</td>
                    <td class="num-cell">${segment.num_ksi_crashes}</td>
                    <td class="num-cell">${segment.fatality_count}</td>
                    <td class="num-cell">${segment.crash_density.toFixed(4)}</td>
                    <td class="coordinates-info">${segment.coordinates.length} points</td>
                `;
                tbody.appendChild(row);
            });
            
            document.getElementById('tableContainer').style.display = 'block';
            
            // Update sorted column indicator
            const sortBy = document.getElementById('sortBy').value;
            document.querySelectorAll('th').forEach(th => {
                th.classList.remove('sorted');
                if (th.getAttribute('data-sort') === sortBy) {
                    th.classList.add('sorted');
                }
            });
        }
        
        function updateStats(result) {
            const statsContainer = document.getElementById('stats');
            const riskCounts = { high: 0, medium: 0, low: 0 };
            let totalCrashes = 0;
            
            result.data.forEach(segment => {
                riskCounts[segment.risk_label]++;
                totalCrashes += segment.num_total_crashes;
            });
            
            // If we have pagination info, show filtered stats
            // For now, we'll just show current page stats
            statsContainer.innerHTML = `
                <div class="stat-card">
                    <div class="stat-label">Segments Shown</div>
                    <div class="stat-value">${result.data.length}</div>
                </div>
                <div class="stat-card high-risk">
                    <div class="stat-label">High Risk</div>
                    <div class="stat-value">${riskCounts.high}</div>
                </div>
                <div class="stat-card medium-risk">
                    <div class="stat-label">Medium Risk</div>
                    <div class="stat-value">${riskCounts.medium}</div>
                </div>
                <div class="stat-card low-risk">
                    <div class="stat-label">Low Risk</div>
                    <div class="stat-value">${riskCounts.low}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Total Crashes</div>
                    <div class="stat-value">${totalCrashes}</div>
                </div>
            `;
        }
        
        function updatePagination(pagination) {
            document.getElementById('paginationInfo').textContent = 
                `Showing ${((pagination.page - 1) * pagination.per_page) + 1} - ${Math.min(pagination.page * pagination.per_page, pagination.total)} of ${pagination.total} segments`;
            
            document.getElementById('prevPage').disabled = pagination.page === 1;
            document.getElementById('nextPage').disabled = pagination.page >= pagination.total_pages;
            
            document.getElementById('pagination').style.display = 'flex';
        }
        
        async function showDataVerification() {
            const detailsDiv = document.getElementById('verificationDetails');
            detailsDiv.style.display = 'block';
            detailsDiv.innerHTML = '<p>Loading verification data...</p>';
            
            try {
                const response = await fetch('/api/data-verification');
                const result = await response.json();
                
                if (!response.ok) {
                    throw new Error(result.error || 'Failed to load verification data');
                }
                
                let html = '<h4 style="margin-top: 0;">Data Summary</h4>';
                html += `<table style="width: 100%; border-collapse: collapse; margin-bottom: 15px;">`;
                html += `<tr><td style="padding: 5px;"><strong>Total Segments:</strong></td><td style="padding: 5px;">${result.summary.total_segments.toLocaleString()}</td></tr>`;
                html += `<tr><td style="padding: 5px;"><strong>Total Crashes:</strong></td><td style="padding: 5px;">${result.summary.total_crashes.toLocaleString()}</td></tr>`;
                html += `<tr><td style="padding: 5px;"><strong>Total KSI Crashes:</strong></td><td style="padding: 5px;">${result.summary.total_ksi_crashes.toLocaleString()}</td></tr>`;
                html += `<tr><td style="padding: 5px;"><strong>KSI Ratio:</strong></td><td style="padding: 5px;">${result.summary.ksi_ratio}%</td></tr>`;
                html += `<tr><td style="padding: 5px;"><strong>Total Fatalities:</strong></td><td style="padding: 5px;">${result.summary.total_fatalities.toLocaleString()}</td></tr>`;
                if (result.summary.segments_with_fatalities !== undefined) {
                    html += `<tr><td style="padding: 5px;"><strong>Segments with Fatalities:</strong></td><td style="padding: 5px;">${result.summary.segments_with_fatalities.toLocaleString()}</td></tr>`;
                }
                html += `</table>`;
                
                // Show fatality accuracy check
                if (result.fatality_accuracy_check) {
                    const acc = result.fatality_accuracy_check;
                    const statusColor = acc.likely_accurate ? '#28a745' : '#dc3545';
                    const statusText = acc.likely_accurate ? '✅ Likely Accurate' : '❌ Likely Inaccurate';
                    html += `<div style="background: ${acc.likely_accurate ? '#d4edda' : '#f8d7da'}; border: 1px solid ${statusColor}; border-radius: 4px; padding: 10px; margin-bottom: 15px;">`;
                    html += `<h5 style="margin-top: 0; color: ${statusColor};">Fatality Data Accuracy: ${statusText}</h5>`;
                    html += `<p style="margin-bottom: 5px;"><strong>Total Fatalities in Data:</strong> ${acc.total_fatalities_in_data.toLocaleString()}</p>`;
                    html += `<p style="margin-bottom: 5px;"><strong>Expected Range:</strong> ${acc.expected_range}</p>`;
                    if (!acc.likely_accurate) {
                        html += `<p style="margin-bottom: 0; color: ${statusColor};"><strong>⚠️ Warning:</strong> ${acc.warning}</p>`;
                    }
                    html += `</div>`;
                }
                
                html += `<h4>Data Structure Explanation</h4>`;
                html += `<p><strong>Total Crashes Source:</strong> ${result.data_structure_explanation.total_crashes_source}</p>`;
                html += `<p><strong>KSI Crashes Source:</strong> ${result.data_structure_explanation.ksi_crashes_source}</p>`;
                html += `<p><em>${result.data_structure_explanation.note}</em></p>`;
                
                if (result.inconsistencies.count > 0) {
                    html += `<h4 style="color: #dc3545;">⚠️ Data Inconsistencies Found (${result.inconsistencies.count})</h4>`;
                    html += `<p><em>${result.inconsistencies.note}</em></p>`;
                    html += `<ul>`;
                    result.inconsistencies.examples.forEach(seg => {
                        html += `<li><strong>${seg.street}</strong> (ID: ${seg.id}): ${seg.ksi_crashes} KSI crashes but only ${seg.total_crashes} total crashes</li>`;
                    });
                    html += `</ul>`;
                }
                
                if (result.sample_high_crash_low_ksi.count > 0) {
                    html += `<h4>Sample Segments: High Crashes, Low/No KSI</h4>`;
                    html += `<p><em>${result.sample_high_crash_low_ksi.note}</em></p>`;
                    html += `<table style="width: 100%; border-collapse: collapse; font-size: 12px;">`;
                    html += `<tr style="background: #f5f5f7;"><th style="padding: 8px; text-align: left;">Street</th><th style="padding: 8px;">Total Crashes</th><th style="padding: 8px;">KSI Crashes</th><th style="padding: 8px;">Fatalities</th></tr>`;
                    result.sample_high_crash_low_ksi.examples.forEach(seg => {
                        html += `<tr><td style="padding: 8px;">${seg.street}</td><td style="padding: 8px; text-align: center;">${seg.total_crashes}</td><td style="padding: 8px; text-align: center;">${seg.ksi_crashes}</td><td style="padding: 8px; text-align: center;">${seg.fatalities}</td></tr>`;
                    });
                    html += `</table>`;
                }
                
                detailsDiv.innerHTML = html;
            } catch (error) {
                detailsDiv.innerHTML = `<p style="color: #dc3545;">Error loading verification data: ${error.message}</p>`;
            }
        }
        
        async function showFatalityDiagnostic() {
            const detailsDiv = document.getElementById('fatalityDiagnosticDetails');
            detailsDiv.style.display = 'block';
            detailsDiv.innerHTML = '<p>Analyzing fatality data structure...</p>';
            
            try {
                const response = await fetch('/api/fatality-diagnostic');
                const result = await response.json();
                
                if (!response.ok) {
                    throw new Error(result.error || 'Failed to load diagnostic data');
                }
                
                let html = '<h4 style="margin-top: 0;">🔍 Fatality Data Diagnostic</h4>';
                
                if (result.error) {
                    html += `<p style="color: #dc3545;"><strong>Error:</strong> ${result.error}</p>`;
                    if (result.traceback) {
                        html += `<pre style="background: #f5f5f5; padding: 10px; border-radius: 4px; overflow-x: auto; font-size: 11px;">${result.traceback}</pre>`;
                    }
                } else {
                    // Collision data info
                    html += '<h5>📊 Collision Dataset</h5>';
                    if (result.collision_data.error) {
                        html += `<p style="color: #dc3545;">Error: ${result.collision_data.error}</p>`;
                    } else {
                        html += `<ul>`;
                        html += `<li><strong>Total Records:</strong> ${result.collision_data.total_records?.toLocaleString() || 'N/A'}</li>`;
                        html += `<li><strong>Expected Fatalities Column:</strong> <code>${result.collision_data.fatalities_column}</code></li>`;
                        html += `<li><strong>Column Exists:</strong> ${result.collision_data.has_fatalities_col ? '✅ Yes' : '❌ No'}</li>`;
                        html += `<li><strong>Expected Injury Column:</strong> <code>${result.collision_data.injury_column}</code></li>`;
                        html += `<li><strong>Column Exists:</strong> ${result.collision_data.has_injury_col ? '✅ Yes' : '❌ No'}</li>`;
                        if (result.collision_data.fatalities_column_stats) {
                            html += `<li><strong>Non-Zero Fatalities:</strong> ${result.collision_data.non_zero_fatalities_count || 0}</li>`;
                            html += `<li><strong>Unique Values in Fatalities Column:</strong> ${Object.keys(result.collision_data.fatalities_column_stats.unique_values || {}).length}</li>`;
                        }
                        if (result.collision_data.fatal_in_injury_count !== undefined) {
                            html += `<li><strong>Records with "Fatal" in Injury Column:</strong> ${result.collision_data.fatal_in_injury_count}</li>`;
                        }
                        html += `</ul>`;
                        
                        if (result.collision_data.fatalities_column_stats) {
                            html += `<details style="margin-top: 10px;"><summary>Sample Fatalities Column Values</summary>`;
                            html += `<pre style="background: #f5f5f5; padding: 10px; border-radius: 4px; overflow-x: auto; font-size: 11px;">${JSON.stringify(result.collision_data.fatalities_column_stats, null, 2)}</pre>`;
                            html += `</details>`;
                        }
                    }
                    
                    // KSI data info
                    html += '<h5 style="margin-top: 20px;">🚨 KSI Dataset</h5>';
                    if (result.ksi_data.error) {
                        html += `<p style="color: #dc3545;">Error: ${result.ksi_data.error}</p>`;
                    } else {
                        html += `<ul>`;
                        html += `<li><strong>Total Records:</strong> ${result.ksi_data.total_records?.toLocaleString() || 'N/A'}</li>`;
                        html += `<li><strong>Expected Fatalities Column:</strong> <code>${result.ksi_data.fatalities_column}</code></li>`;
                        html += `<li><strong>Column Exists:</strong> ${result.ksi_data.has_fatalities_col ? '✅ Yes' : '❌ No'}</li>`;
                        html += `<li><strong>Expected Injury Column:</strong> <code>${result.ksi_data.injury_column}</code></li>`;
                        html += `<li><strong>Column Exists:</strong> ${result.ksi_data.has_injury_col ? '✅ Yes' : '❌ No'}</li>`;
                        if (result.ksi_data.fatalities_column_stats) {
                            html += `<li><strong>Non-Zero Fatalities:</strong> ${result.ksi_data.non_zero_fatalities_count || 0}</li>`;
                            html += `<li><strong>Data Type:</strong> ${result.ksi_data.fatalities_column_stats.data_type || 'N/A'}</li>`;
                        }
                        if (result.ksi_data.fatal_in_injury_count !== undefined) {
                            html += `<li><strong>Records with "Fatal" in Injury Column:</strong> ${result.ksi_data.fatal_in_injury_count}</li>`;
                        }
                        html += `</ul>`;
                        
                        if (result.ksi_data.fatalities_column_stats) {
                            html += `<details style="margin-top: 10px;"><summary>Sample Fatalities Column Values</summary>`;
                            html += `<pre style="background: #f5f5f5; padding: 10px; border-radius: 4px; overflow-x: auto; font-size: 11px;">${JSON.stringify(result.ksi_data.fatalities_column_stats, null, 2)}</pre>`;
                            html += `</details>`;
                        }
                    }
                    
                    // All available columns
                    if (result.collision_data.columns) {
                        html += `<details style="margin-top: 15px;"><summary>All Columns in Collision Dataset</summary>`;
                        html += `<p style="font-size: 12px;">${result.collision_data.columns.join(', ')}</p>`;
                        html += `</details>`;
                    }
                    if (result.ksi_data.columns) {
                        html += `<details style="margin-top: 10px;"><summary>All Columns in KSI Dataset</summary>`;
                        html += `<p style="font-size: 12px;">${result.ksi_data.columns.join(', ')}</p>`;
                        html += `</details>`;
                    }
                }
                
                detailsDiv.innerHTML = html;
            } catch (error) {
                detailsDiv.innerHTML = `<p style="color: #dc3545;">Error loading diagnostic data: ${error.message}</p>`;
            }
        }
        
        async function showModelFeatures() {
            const detailsDiv = document.getElementById('modelFeaturesDetails');
            detailsDiv.style.display = 'block';
            detailsDiv.innerHTML = '<p>Loading model features...</p>';
            
            try {
                const response = await fetch('/api/model-features');
                const result = await response.json();
                
                if (!response.ok) {
                    throw new Error(result.error || 'Failed to load model features');
                }
                
                let html = '<h4 style="margin-top: 0;">📊 Model Input Features</h4>';
                html += `<p><strong>Total Features Used:</strong> ${result.total_features}</p>`;
                
                // Feature categories
                html += '<h5 style="margin-top: 15px;">Feature Categories</h5>';
                
                if (result.feature_categories.temporal_features.length > 0) {
                    html += '<details style="margin-top: 10px;"><summary><strong>⏰ Temporal Features</strong> (' + result.feature_categories.temporal_features.length + ')</summary>';
                    html += '<ul style="margin-top: 5px;">';
                    result.feature_categories.temporal_features.forEach(f => {
                        html += `<li><code>${f}</code></li>`;
                    });
                    html += '</ul></details>';
                }
                
                if (result.feature_categories.road_characteristics.length > 0) {
                    html += '<details style="margin-top: 10px;"><summary><strong>🛣️ Road Characteristics</strong> (' + result.feature_categories.road_characteristics.length + ')</summary>';
                    html += '<ul style="margin-top: 5px;">';
                    result.feature_categories.road_characteristics.forEach(f => {
                        html += `<li><code>${f}</code></li>`;
                    });
                    html += '</ul></details>';
                }
                
                if (result.feature_categories.other_features.length > 0) {
                    html += '<details style="margin-top: 10px;"><summary><strong>📈 Other Features</strong> (' + result.feature_categories.other_features.length + ')</summary>';
                    html += '<ul style="margin-top: 5px;">';
                    result.feature_categories.other_features.forEach(f => {
                        html += `<li><code>${f}</code></li>`;
                    });
                    html += '</ul></details>';
                }
                
                // Feature importance
                if (result.feature_importance && Object.keys(result.feature_importance).length > 0) {
                    html += '<h5 style="margin-top: 20px;">Feature Importance (Top 15)</h5>';
                    html += '<p style="font-size: 12px; color: #6e6e73;">Higher values indicate features that are more important for predictions</p>';
                    html += '<table style="width: 100%; border-collapse: collapse; font-size: 12px; margin-top: 10px;">';
                    html += '<tr style="background: #f5f5f7;"><th style="padding: 8px; text-align: left;">Feature</th><th style="padding: 8px; text-align: right;">Importance</th></tr>';
                    const topFeatures = Object.entries(result.feature_importance).slice(0, 15);
                    topFeatures.forEach(([feature, importance]) => {
                        const percentage = (importance * 100).toFixed(2);
                        html += `<tr><td style="padding: 8px;"><code>${feature}</code></td><td style="padding: 8px; text-align: right;">${percentage}%</td></tr>`;
                    });
                    html += '</table>';
                }
                
                // Excluded columns explanation
                html += '<h5 style="margin-top: 20px;">🚫 Excluded Features (Data Leakage Prevention)</h5>';
                html += '<p style="font-size: 12px; color: #6e6e73;">' + result.excluded_columns.note + '</p>';
                html += '<details style="margin-top: 10px;"><summary><strong>Excluded Columns</strong></summary>';
                html += '<ul style="margin-top: 5px; font-size: 11px;">';
                result.excluded_columns.data_leakage.forEach(col => {
                    html += `<li><code>${col}</code></li>`;
                });
                html += '</ul></details>';
                
                // All features list
                html += '<details style="margin-top: 15px;"><summary><strong>Complete Feature List</strong></summary>';
                html += '<p style="font-size: 11px; font-family: monospace; background: #f5f5f7; padding: 10px; border-radius: 4px; overflow-x: auto;">';
                html += result.feature_columns.join(', ');
                html += '</p></details>';
                
                detailsDiv.innerHTML = html;
            } catch (error) {
                detailsDiv.innerHTML = `<p style="color: #dc3545;">Error loading model features: ${error.message}</p>`;
            }
        }
    </script>
</body>
</html>
    """


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app.run(host="0.0.0.0", port=8000, debug=True, use_reloader=False, threaded=False)
