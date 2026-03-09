"""
flask API server
serves risk predictions from the trained model
"""

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import logging
import sys
from datetime import datetime
from shapely.geometry import Point, box

# import existing modules
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# model & pipeline imports
from src.models.model_trainer import TemporalCountModelTrainer  # type: ignore
from src.data_processing.data_loader import (
    load_road_network,
    load_model_dataset,
    merge_model_dataset_into_road_network,
    load_historical_weather,
)
from src.data_processing.spatial_join_fast import _ensure_stable_segment_id  # type: ignore
from src.feature_engineering.panel_builder import (  # type: ignore
    PanelConfig,
    build_inference_panel_for_datetime,
)
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
TEMPORAL_MODEL_PATH = (
    PROJECT_ROOT / "outputs" / "models" / "toronto_temporal_count_model.pkl"
)
temporal_trainer: Optional[TemporalCountModelTrainer] = None

try:
    temporal_trainer = TemporalCountModelTrainer()
    temporal_trainer.load_model(str(TEMPORAL_MODEL_PATH))
    logging.info("Temporal count model loaded successfully")
except Exception as e:
    logging.warning(f"Failed to load temporal count model: {e}")
    temporal_trainer = None

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
_lam_values_sorted = None  # for percentile-based display confidence
crash_count_latest = None  # segment_id -> crash count in latest window

try:
    data_dir = PROJECT_ROOT / "data"
    road_network = load_road_network(data_dir)
    logging.info(f"Loaded {len(road_network)} road segments")

    # Ensure stable segment_id and merge ADT/speed for on-demand inference
    road_network = _ensure_stable_segment_id(road_network)
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

# For on-demand datetime-based inference
crash_counts_sparse: Optional[pd.DataFrame] = None
weather_data: Optional[pd.DataFrame] = None
_lambda_cache: Dict[str, Dict[str, Any]] = {}
CRASH_COUNTS_PATH = PROJECT_ROOT / "outputs" / "reports" / "crash_counts_sparse.parquet"

try:
    if CRASH_COUNTS_PATH.exists():
        crash_counts_sparse = pd.read_parquet(CRASH_COUNTS_PATH)
        crash_counts_sparse["window_start"] = pd.to_datetime(
            crash_counts_sparse["window_start"]
        )
        logging.info(
            "Loaded crash counts sparse: %d rows", len(crash_counts_sparse)
        )
except Exception as e:
    logging.warning(f"Could not load crash counts sparse: {e}")

try:
    weather_data = load_historical_weather(PROJECT_ROOT / "data")
    if weather_data is not None:
        logging.info("Loaded historical weather for on-demand inference")
except Exception as e:
    logging.warning(f"Could not load weather data: {e}")

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


def _lambda_to_risk_label(
    lam: float,
    p70: Optional[float] = None,
    p90: Optional[float] = None,
) -> str:
    """
    Map λ (crashes per hour) to risk_label (low/medium/high) using percentile thresholds.
    """
    global _lambda_p70, _lambda_p90
    p70 = p70 if p70 is not None else _lambda_p70
    p90 = p90 if p90 is not None else _lambda_p90
    if p70 is None or p90 is None:
        return "low"  # fallback if thresholds not yet computed
    if lam <= p70:
        return "low"
    if lam <= p90:
        return "medium"
    return "high"


def _normalize_segment_id(seg_id) -> Optional[int]:
    """Normalize segment_id to int for consistent dict lookups (handles float from GeoJSON)."""
    if seg_id is None or (isinstance(seg_id, float) and np.isnan(seg_id)):
        return None
    try:
        if isinstance(seg_id, (int, np.integer)):
            return int(seg_id)
        f = float(seg_id)
        return None if np.isnan(f) else int(f)
    except (ValueError, TypeError):
        return None


def _lambda_to_display_confidence(
    lam: float, lam_values_sorted: Optional[np.ndarray] = None
) -> float:
    """
    Map λ to a 0-1 display confidence (percentile rank).
    """
    global _lam_values_sorted
    arr = lam_values_sorted if lam_values_sorted is not None else _lam_values_sorted
    if arr is None or len(arr) == 0:
        return 0.5
    if np.isnan(lam) or np.isinf(lam) or lam < 0:
        return 0.5
    # If all λ are identical, percentile is degenerate
    if float(arr[-1]) <= float(arr[0]):
        return 0.5
    # count of values strictly less than lam
    rank = int(np.searchsorted(arr, lam, side="left"))
    n = len(arr)
    # avoid 100% for everyone: use (rank + 0.5) / (n + 1) for smoother 0-1 spread
    pct = (rank + 0.5) / (n + 1)
    return float(np.clip(pct, 0.0, 1.0))


def _compute_lambda_map_for_latest_window():
    """
    Compute λ_per_hour for each segment in the most recent panel window.

    This is used to annotate routing edges with expected crashes and to derive
    risk_label for the iOS app (temporal model replaces classification model).
    """
    global lambda_per_hour_latest, latest_window_start, _lambda_p70, _lambda_p90, _lam_values_sorted, crash_count_latest

    if temporal_trainer is None or temporal_trainer.model is None:
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

    # Map segment_id -> λ_per_hour and crash_count (normalize keys to int for GeoJSON float compatibility)
    segment_ids_raw = current_slice["segment_id"].values
    lambda_per_hour_latest = {}
    crash_count_latest = {}
    for i, sid in enumerate(segment_ids_raw):
        nid = _normalize_segment_id(sid)
        if nid is not None:
            lam_val = float(lambda_per_hour[i])
            lambda_per_hour_latest[nid] = lam_val
            if sid != nid:
                lambda_per_hour_latest[sid] = lam_val
            val = current_slice.iloc[i].get("crash_count", 0)
            crash_count_latest[nid] = int(val) if pd.notna(val) else 0
            if sid != nid:
                crash_count_latest[sid] = crash_count_latest[nid]

    # Compute percentile thresholds for λ → risk_label mapping (low ≤ p70, medium ≤ p90, high > p90)
    lam_values = np.array(list(lambda_per_hour_latest.values()), dtype=float)
    _lambda_p70 = float(np.percentile(lam_values, 70))
    _lambda_p90 = float(np.percentile(lam_values, 90))
    _lam_values_sorted = np.sort(lam_values)
    logging.info(
        "Computed λ_per_hour for latest window %s for %d segments (p70=%.6f, p90=%.6f).",
        latest_window_start,
        len(lambda_per_hour_latest),
        _lambda_p70,
        _lambda_p90,
    )


def _get_or_compute_lambda_map(as_of: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Get lambda map for a given datetime, using cache when possible.
    Returns dict with: lambda_per_hour, crash_count, lam_values_sorted, p70, p90, panel_slice.
    Falls back to default (panel_latest) when as_of is None or on-demand is unavailable.
    """
    if as_of is None or as_of.strip() == "":
        if lambda_per_hour_latest is None:
            _compute_lambda_map_for_latest_window()
        return {
            "lambda_per_hour": lambda_per_hour_latest,
            "crash_count": crash_count_latest,
            "lam_values_sorted": _lam_values_sorted,
            "p70": _lambda_p70,
            "p90": _lambda_p90,
            "panel_slice": None,
        }

    if (
        crash_counts_sparse is None
        or crash_counts_sparse.empty
        or temporal_trainer is None
        or road_network is None
    ):
        if lambda_per_hour_latest is None:
            _compute_lambda_map_for_latest_window()
        return {
            "lambda_per_hour": lambda_per_hour_latest,
            "crash_count": crash_count_latest,
            "lam_values_sorted": _lam_values_sorted,
            "p70": _lambda_p70,
            "p90": _lambda_p90,
            "panel_slice": None,
        }

    try:
        target_ts = pd.Timestamp(as_of).tz_localize(None)
    except Exception:
        if lambda_per_hour_latest is None:
            _compute_lambda_map_for_latest_window()
        return {
            "lambda_per_hour": lambda_per_hour_latest,
            "crash_count": crash_count_latest,
            "lam_values_sorted": _lam_values_sorted,
            "p70": _lambda_p70,
            "p90": _lambda_p90,
            "panel_slice": None,
        }

    cache_key = target_ts.floor("H").isoformat()
    if cache_key in _lambda_cache:
        return _lambda_cache[cache_key]

    config = PanelConfig(window_size_hours=1, horizon_hours=1)
    panel = build_inference_panel_for_datetime(
        crash_counts_sparse=crash_counts_sparse,
        road_network=road_network,
        weather_data=weather_data,
        target_datetime=target_ts,
        config=config,
    )
    X_current, _ = temporal_trainer.prepare_panel_features(panel)
    lambda_window = temporal_trainer.predict_lambda(X_current)
    window_size_hours = config.window_size_hours
    lambda_per_hour = lambda_window / float(window_size_hours)

    segment_ids_raw = panel["segment_id"].values
    lam_map = {}
    crash_map = {}
    for i, sid in enumerate(segment_ids_raw):
        nid = _normalize_segment_id(sid)
        if nid is not None:
            lam_val = float(lambda_per_hour[i])
            lam_map[nid] = lam_val
            if sid != nid:
                lam_map[sid] = lam_val
            val = panel.iloc[i].get("crash_count", 0)
            crash_map[nid] = int(val) if pd.notna(val) else 0
            if sid != nid:
                crash_map[sid] = crash_map[nid]

    lam_values = np.array(list(lam_map.values()), dtype=float)
    p70 = float(np.percentile(lam_values, 70))
    p90 = float(np.percentile(lam_values, 90))
    lam_sorted = np.sort(lam_values)

    result = {
        "lambda_per_hour": lam_map,
        "crash_count": crash_map,
        "lam_values_sorted": lam_sorted,
        "p70": p70,
        "p90": p90,
        "panel_slice": panel,
    }
    _lambda_cache[cache_key] = result
    if len(_lambda_cache) > 48:
        oldest = min(_lambda_cache.keys())
        del _lambda_cache[oldest]
    return result


def _get_risk_driver_features_for_segment(
    segment_id, panel_slice: Optional[pd.DataFrame] = None
):
    """
    Extract a compact set of 'risk driver' features for a segment.
    Uses panel_slice if provided (for on-demand inference), else panel_data.
    """
    if panel_slice is not None and not panel_slice.empty:
        match = panel_slice[panel_slice["segment_id"] == segment_id]
        if match.empty:
            match = panel_slice[
                panel_slice["segment_id"].astype(str) == str(segment_id)
            ]
        if match.empty:
            return {}
        row = match.iloc[0]
    elif panel_data is not None and not panel_data.empty and latest_window_start is not None:
        row_match = panel_data[
            (panel_data["segment_id"] == segment_id)
            & (panel_data["window_start"] == latest_window_start)
        ]
        if row_match.empty:
            return {}
        row = row_match.iloc[0]
    else:
        return {}
    keys = [
        # static segment features
        "is_oneway",
        "from_intersection_degree",
        "to_intersection_degree",
        # temporal context
        "hour_of_day",
        "day_of_week",
        "is_weekend",
        "month",
        "season",
        # crash history (weekly-spaced when using weekly windows)
        "crashes_1_week_ago",
        "crashes_2_weeks_ago",
        "crashes_4_weeks_ago",
        "rolling_mean_4_weeks",
        "rolling_max_4_weeks",
        # finer windows (hourly)
        "past_crash_count_1h",
        "past_crash_count_24h",
        "past_crash_count_7d",
        "rolling_mean_24h",
        "rolling_max_24h",
        "rolling_mean_7d",
        "rolling_max_30d",
        # weather (if present)
        "temperature",
        "precipitation",
        "snow_mm",
        "visibility",
        "wind_speed",
        "weather_condition",
        "is_missing_weather",
    ]
    drivers = {}
    for k in keys:
        if k in row.index:
            val = row[k]
            if isinstance(val, (np.integer, np.int64, np.int32)):
                drivers[k] = int(val)
            elif isinstance(val, (np.floating, np.float64, np.float32)):
                drivers[k] = float(val)
            else:
                drivers[k] = val
    return drivers


def _calculate_weather_risk_multiplier(weather_data: Optional[dict]) -> float:
    """
    Research-backed risk multiplier from peer-reviewed studies.
    Sources: ETRR 2022, Nature Scientific Reports 2025, Accident Analysis & Prevention.
    See WEATHER_MULTIPLIERS_CITATIONS.md for references.
    """
    if not weather_data:
        return 1.0
    condition = (weather_data.get("condition") or "clear").lower()
    visibility = weather_data.get("visibility")
    precipitation = weather_data.get("precipitation") or 0

    multipliers = {
        "clear": 1.0,
        "cloudy": 1.02,
        "mist": 1.15,
        "rain": 1.35,
        "heavy_rain": 1.45,
        "snow": 1.6,
        "heavy_snow": 2.0,
        "fog": 2.5,
        "thunderstorm": 1.6,
        "sleet": 1.9,
    }
    mult = multipliers.get(condition, 1.0)
    if visibility is not None:
        if visibility < 1.0:
            mult *= 1.3
        elif visibility < 3.0:
            mult *= 1.2
        elif visibility < 5.0:
            mult *= 1.1
    if precipitation and float(precipitation) > 5.0:
        mult *= 1.1
    return mult


def _calculate_time_risk_multiplier(time_data: Optional[dict]) -> float:
    """Time-of-day risk multiplier (rush hour, night)."""
    if not time_data:
        return 1.0
    hour = time_data.get("hour")
    is_weekend = time_data.get("is_weekend", False)
    if hour is None:
        return 1.0
    h = int(hour)
    if h >= 23 or h < 5:
        return 1.4
    if h >= 22 or h < 6:
        return 1.25
    if (7 <= h <= 9) or (17 <= h <= 19):
        return 1.3 if not is_weekend else 1.1
    if 9 <= h < 17:
        return 1.0
    return 1.1


def _adjust_risk_for_conditions(
    risk_label: str, risk_score: int, weather_mult: float, time_mult: float
) -> tuple[str, int]:
    """
    Adjust risk_label and risk_score based on weather and time multipliers.
    Returns (adjusted_risk_label, adjusted_risk_score).
    """
    risk_values = {"low": 1, "medium": 2, "high": 3}
    base = risk_values.get(risk_label, 1)
    combined = weather_mult * time_mult
    adj_value = base * combined

    if adj_value >= 2.5:
        new_label = "high"
    elif adj_value >= 1.5:
        new_label = "medium"
    else:
        new_label = "low"

    adj_score = min(100, int(risk_score * combined))
    return new_label, adj_score


def _build_risk_explanation(drivers: dict, risk_label: str, risk_score: int) -> str:
    """Build human-readable explanation from risk drivers and score."""
    parts = []
    if risk_label == "high":
        parts.append(
            "This segment has elevated crash risk based on current conditions."
        )
    elif risk_label == "medium":
        parts.append("This segment has moderate crash risk.")
    else:
        parts.append("This segment has lower crash risk relative to others.")

    if drivers:
        factors = []
        hour = drivers.get("hour_of_day")
        if hour is not None and ((7 <= hour <= 9) or (16 <= hour <= 18)):
            factors.append("rush hour")
        if drivers.get("is_weekend"):
            factors.append("weekend traffic")
        past_24h = drivers.get("past_crash_count_24h") or drivers.get(
            "crashes_1_week_ago"
        )
        if past_24h and int(past_24h) > 0:
            factors.append("recent crash activity")
        if (
            drivers.get("precipitation", 0)
            and float(drivers.get("precipitation", 0)) > 0
        ):
            factors.append("precipitation")
        if drivers.get("temperature") is not None:
            temp = float(drivers["temperature"])
            if temp < 0:
                factors.append("freezing conditions")
        if factors:
            parts.append(" Contributing factors: " + ", ".join(factors) + ".")
        else:
            parts.append(
                " The prediction uses road type, time of day, and crash history."
            )

    parts.append(f" Risk score: {risk_score}/100 (percentile vs. all segments).")
    return "".join(parts)


@app.route("/api/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify(
        {
            "status": "healthy",
            "temporal_model_loaded": temporal_trainer is not None
            and temporal_trainer.model is not None,  # type: ignore[truthy-function]
            "road_network_loaded": road_network is not None,
            "routing_graph_built": road_graph is not None and node_coords is not None,
            "panel_loaded": panel_data is not None,
            "road_segments": len(road_network) if road_network is not None else 0,
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
        return (
            jsonify(
                {
                    "error": "Road network not loaded",
                    "hint": "Check data/centreline geojson exists",
                }
            ),
            500,
        )
    if temporal_trainer is None or temporal_trainer.model is None:
        return (
            jsonify(
                {
                    "error": "Temporal model not loaded",
                    "hint": "Run: python train_temporal_model.py (ensure scikit-learn matches)",
                }
            ),
            500,
        )
    if (
        (panel_data is None or panel_data.empty)
        and (crash_counts_sparse is None or crash_counts_sparse.empty)
    ):
        return (
            jsonify(
                {
                    "error": "Panel data or crash counts not loaded",
                    "hint": "Run: python train_temporal_model.py and pip install pyarrow",
                }
            ),
            500,
        )

    try:
        data = request.get_json() or {}
        north = data.get("north")
        south = data.get("south")
        east = data.get("east")
        west = data.get("west")
        as_of = data.get("as_of")
        weather_data = data.get("weather")
        time_data = data.get("time_of_day")

        weather_mult = _calculate_weather_risk_multiplier(weather_data)
        time_mult = _calculate_time_risk_multiplier(time_data)

        lam_result = _get_or_compute_lambda_map(as_of)
        if lam_result is None:
            return jsonify({"error": "Could not compute risk"}), 500

        lam_map = lam_result["lambda_per_hour"]
        crash_map = lam_result["crash_count"]
        lam_sorted = lam_result["lam_values_sorted"]
        p70, p90 = lam_result["p70"], lam_result["p90"]
        panel_slice = lam_result.get("panel_slice")

        def _get_risk_for_row(row):
            seg_id = row.get("segment_id") or row.get("CENTRELINE_ID", row.name)
            nid = _normalize_segment_id(seg_id)
            lam = lam_map.get(nid, 0.0) if nid is not None else 0.0
            return _lambda_to_risk_label(lam, p70=p70, p90=p90)

        bbox = box(west, south, east, north)
        segments_in_bbox = road_network[road_network.geometry.intersects(bbox)].copy()
        segments_in_bbox["risk_label"] = segments_in_bbox.apply(
            _get_risk_for_row, axis=1
        )

        if len(segments_in_bbox) > 500:
            risk_priority = {"high": 3, "medium": 2, "low": 1}
            segments_in_bbox["_risk_priority"] = segments_in_bbox["risk_label"].map(
                risk_priority
            )
            segments_in_bbox = segments_in_bbox.sort_values(
                "_risk_priority", ascending=False
            ).head(500)

        results = []
        for idx, segment in segments_in_bbox.iterrows():
            coords = _extract_coordinates(segment.geometry)
            risk_label = segment["risk_label"]
            seg_id = segment.get("segment_id") or segment.get("CENTRELINE_ID", idx)
            nid = _normalize_segment_id(seg_id)
            lam = lam_map.get(nid, 0.0) if nid is not None else 0.0
            confidence = _lambda_to_display_confidence(lam, lam_values_sorted=lam_sorted)
            risk_score = int(round(confidence * 100))
            risk_label, risk_score = _adjust_risk_for_conditions(
                risk_label, risk_score, weather_mult, time_mult
            )
            recent_crashes = (
                int(crash_map.get(nid, 0)) if crash_map and nid is not None else 0
            )
            drivers = _get_risk_driver_features_for_segment(
                nid if nid is not None else seg_id,
                panel_slice=panel_slice,
            )
            risk_explanation = _build_risk_explanation(drivers, risk_label, risk_score)

            result = {
                "id": str(seg_id),
                "LINEAR_NAME": segment.get("LINEAR_NAME", "Unknown"),
                "ROAD_CLASS": segment.get("ROAD_CLASS", "Unknown"),
                "segment_length": float(segment.get("segment_length", 0)),
                "risk_label": risk_label,
                "risk_score": risk_score,
                "confidence": confidence,
                "risk_explanation": risk_explanation,
                "num_total_crashes": recent_crashes,
                "num_ksi_crashes": int(segment.get("num_ksi_crashes", 0)),
                "fatality_count": int(segment.get("fatality_count", 0)),
                "coordinates": coords[:50],
            }
            results.append(result)

        logging.info(f"Returning {len(results)} segments for bbox")
        return jsonify(results)

    except Exception as e:
        logging.error(f"Error in risk predictions: {e}")
        import traceback

        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


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
    if temporal_trainer is None or temporal_trainer.model is None:
        return jsonify({"error": "Temporal model not loaded"}), 500
    if (
        (panel_data is None or panel_data.empty)
        and (crash_counts_sparse is None or crash_counts_sparse.empty)
    ):
        return jsonify({"error": "Panel data or crash counts not loaded"}), 500

    try:
        data = request.get_json() or {}
        lat = data.get("latitude")
        lon = data.get("longitude")
        as_of = data.get("as_of")
        weather_data = data.get("weather")
        time_data = data.get("time_of_day")

        weather_mult = _calculate_weather_risk_multiplier(weather_data)
        time_mult = _calculate_time_risk_multiplier(time_data)

        lam_result = _get_or_compute_lambda_map(as_of)
        if lam_result is None:
            return jsonify({"error": "Could not compute risk"}), 500

        lam_map = lam_result["lambda_per_hour"]
        crash_map = lam_result["crash_count"]
        lam_sorted = lam_result["lam_values_sorted"]
        p70, p90 = lam_result["p70"], lam_result["p90"]
        panel_slice = lam_result.get("panel_slice")

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

            seg_id = segment.get("segment_id") or segment.get(
                "CENTRELINE_ID", nearest_idx
            )
            nid = _normalize_segment_id(seg_id)
            lam = lam_map.get(nid, 0.0) if nid is not None else 0.0

            risk_label = _lambda_to_risk_label(lam, p70=p70, p90=p90)
            confidence = _lambda_to_display_confidence(lam, lam_values_sorted=lam_sorted)
            risk_score = int(round(confidence * 100))
            risk_label, risk_score = _adjust_risk_for_conditions(
                risk_label, risk_score, weather_mult, time_mult
            )
            recent_crashes = (
                int(crash_map.get(nid, 0)) if crash_map and nid is not None else 0
            )
            drivers = _get_risk_driver_features_for_segment(
                nid if nid is not None else seg_id,
                panel_slice=panel_slice,
            )
            risk_explanation = _build_risk_explanation(drivers, risk_label, risk_score)

            if risk_label == "high":
                probabilities = {"low": 0.1, "medium": 0.1, "high": 0.8}
            elif risk_label == "medium":
                probabilities = {"low": 0.2, "medium": 0.7, "high": 0.1}
            else:
                probabilities = {"low": 0.8, "medium": 0.15, "high": 0.05}

            segment_info = {
                "id": str(seg_id),
                "LINEAR_NAME": segment.get("LINEAR_NAME", "Unknown"),
                "ROAD_CLASS": segment.get("ROAD_CLASS", "Unknown"),
                "segment_length": float(segment.get("segment_length", 0)),
                "risk_score": risk_score,
                "risk_explanation": risk_explanation,
                "num_total_crashes": recent_crashes,
                "num_ksi_crashes": int(segment.get("num_ksi_crashes", 0)),
                "fatality_count": int(segment.get("fatality_count", 0)),
                "coordinates": _extract_coordinates(segment.geometry),
            }

            response = {
                "riskLevel": risk_label,
                "riskScore": risk_score,
                "riskExplanation": risk_explanation,
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
        "beta": float (optional, hours per expected crash)
    }
    """
    if (
        temporal_trainer is None
        or temporal_trainer.model is None  # type: ignore[truthy-function]
        or road_graph is None
        or node_coords is None
    ):
        return (
            jsonify(
                {
                    "error": "Temporal model or routing graph not initialized. Check /api/health."
                }
            ),
            500,
        )

    try:
        data = request.get_json()
        origin = data.get("origin", {})
        destination = data.get("destination", {})
        beta = float(data.get("beta", 0.1))

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

        origin_point = Point(o_lon, o_lat)
        dest_point = Point(d_lon, d_lat)

        # Ensure λ map is ready
        global lambda_per_hour_latest
        if lambda_per_hour_latest is None:
            _compute_lambda_map_for_latest_window()

        # Apply λ to edges as expected_crashes and risk-weight
        apply_risk_to_edge_costs(road_graph, lambda_per_hour_latest, beta_hours_per_expected_crash=beta)  # type: ignore[arg-type]

        # Snap origin/destination to nearest graph nodes
        start_node = snap_to_graph(origin_point, node_coords)  # type: ignore[arg-type]
        end_node = snap_to_graph(dest_point, node_coords)  # type: ignore[arg-type]

        # Find fastest and safer paths
        fastest_path = find_fastest_route(road_graph, start_node, end_node)  # type: ignore[arg-type]
        safer_path = find_safer_route(road_graph, start_node, end_node)  # type: ignore[arg-type]

        fastest_summary = calculate_route_risk(road_graph, fastest_path)  # type: ignore[arg-type]
        safer_summary = calculate_route_risk(road_graph, safer_path)  # type: ignore[arg-type]

        # Collect segments along each path
        fastest_edges = path_edges(road_graph, fastest_path)  # type: ignore[arg-type]
        safer_edges = path_edges(road_graph, safer_path)  # type: ignore[arg-type]

        fastest_segments = {data["segment_id"] for _, _, data in fastest_edges}
        safer_segments = {data["segment_id"] for _, _, data in safer_edges}

        avoided_segments = sorted(fastest_segments - safer_segments)

        # Build risk driver explanations for avoided segments
        avoided_details = []
        for seg_id in avoided_segments:
            nid = _normalize_segment_id(seg_id)
            lam = (
                float(lambda_per_hour_latest.get(nid, 0.0))
                if lambda_per_hour_latest and nid is not None
                else 0.0
            )
            drivers = _get_risk_driver_features_for_segment(
                nid if nid is not None else seg_id
            )
            avoided_details.append(
                {
                    "segmentId": seg_id,
                    "lambdaPerHour": lam,
                    "riskDrivers": drivers,
                }
            )

        response = {
            "fastest": {
                "nodes": fastest_path,
                "segmentIds": list(fastest_segments),
                "summary": {
                    "totalTravelTimeHours": float(
                        fastest_summary["total_travel_time_hours"]
                    ),
                    "expectedCrashes": float(fastest_summary["expected_crashes"]),
                    "routeProbability": float(fastest_summary["route_probability"]),
                },
            },
            "safer": {
                "nodes": safer_path,
                "segmentIds": list(safer_segments),
                "summary": {
                    "totalTravelTimeHours": float(
                        safer_summary["total_travel_time_hours"]
                    ),
                    "expectedCrashes": float(safer_summary["expected_crashes"]),
                    "routeProbability": float(safer_summary["route_probability"]),
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
    if temporal_trainer is None or temporal_trainer.model is None:
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
        model = temporal_trainer.model
        if hasattr(model, "feature_importances_") and feature_columns:
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
                temporal_trainer is not None
                and temporal_trainer.model is not None
                and panel_data is not None
                and not panel_data.empty
            ):
                try:
                    if lambda_per_hour_latest is None:
                        _compute_lambda_map_for_latest_window()
                    seg_id = segment.get("segment_id") or segment.get(
                        "CENTRELINE_ID", idx
                    )
                    nid = _normalize_segment_id(seg_id)
                    lam = lambda_per_hour_latest.get(nid, 0.0) if nid is not None else 0.0  # type: ignore[union-attr]
                    risk_label = _lambda_to_risk_label(lam)
                    confidence = _lambda_to_display_confidence(lam)
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
    # threaded=False avoids GEOS/Shapely double-free on macOS when handling concurrent requests
    app.run(host="0.0.0.0", port=8000, debug=True, use_reloader=False, threaded=False)
