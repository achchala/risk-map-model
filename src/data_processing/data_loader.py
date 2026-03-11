"""
Data loading and cleaning module for Toronto Road Segment Crash Risk Prediction

This module handles loading and initial cleaning of:
1. Traffic collision data (Excel)
2. Killed or Seriously Injured (KSI) data (CSV)
3. Road network geometry (GeoJSON)
"""

import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path
from typing import Optional
import logging
from datetime import datetime
import sys
import os

# Add parent directory to path for config import
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import *

logger = logging.getLogger(__name__)


def load_collision_data(data_dir: Path) -> pd.DataFrame:
    """
    Load and clean traffic collision data from Excel file

    Args:
        data_dir: Path to data directory

    Returns:
        Cleaned collision DataFrame
    """
    logger.info("Loading collision data...")

    file_path = data_dir / COLLISION_DATA_FILE
    if not file_path.exists():
        raise FileNotFoundError(f"Collision data file not found: {file_path}")

    # Load Excel file
    df = pd.read_excel(file_path)
    logger.info(f"Loaded {len(df)} collision records")

    # Basic cleaning
    df = df.copy()

    # Handle missing coordinates - filter out rows without valid lat/lon
    lat_col = COLLISION_COLUMNS["latitude"]
    lon_col = COLLISION_COLUMNS["longitude"]

    initial_count = len(df)
    df = df.dropna(subset=[lat_col, lon_col])
    dropped_count = initial_count - len(df)
    logger.info(f"Dropped {dropped_count} records with missing coordinates")

    # Convert coordinates to numeric
    df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
    df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")

    # Filter out invalid coordinates (outside reasonable bounds for Toronto)
    df = df[
        (df[lat_col] >= 43.5)
        & (df[lat_col] <= 44.0)
        & (df[lon_col] >= -79.8)
        & (df[lon_col] <= -79.0)
    ]
    logger.info(f"After coordinate filtering: {len(df)} records")

    # Convert to GeoDataFrame
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
        crs=f"EPSG:{CRS_EPSG}",
    )

    # Clean date/time columns
    date_col = COLLISION_COLUMNS["date"]
    time_col = COLLISION_COLUMNS["time"]

    if date_col in gdf.columns:
        gdf["DATE"] = pd.to_datetime(gdf[date_col], errors="coerce")
        gdf = gdf.dropna(subset=["DATE"])

    if time_col in gdf.columns:
        # Convert time to hour of day (already numeric for collision data)
        gdf["HOUR"] = pd.to_numeric(gdf[time_col], errors="coerce")
        gdf = gdf.dropna(subset=["HOUR"])

    logger.info(f"Final collision data: {len(gdf)} records")
    return gdf


def load_ksi_data(data_dir: Path) -> pd.DataFrame:
    """
    Load and clean Killed or Seriously Injured (KSI) data

    Args:
        data_dir: Path to data directory

    Returns:
        Cleaned KSI DataFrame
    """
    logger.info("Loading KSI data...")

    file_path = data_dir / KSI_DATA_FILE
    if not file_path.exists():
        raise FileNotFoundError(f"KSI data file not found: {file_path}")

    # Load CSV file
    df = pd.read_csv(file_path)
    logger.info(f"Loaded {len(df)} KSI records")

    # Basic cleaning
    df = df.copy()

    # Handle missing coordinates
    lat_col = KSI_COLUMNS["latitude"]
    lon_col = KSI_COLUMNS["longitude"]

    initial_count = len(df)
    df = df.dropna(subset=[lat_col, lon_col])
    dropped_count = initial_count - len(df)
    logger.info(f"Dropped {dropped_count} KSI records with missing coordinates")

    # Convert coordinates to numeric
    df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
    df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")

    # Filter out invalid coordinates
    df = df[
        (df[lat_col] >= 43.5)
        & (df[lat_col] <= 44.0)
        & (df[lon_col] >= -79.8)
        & (df[lon_col] <= -79.0)
    ]
    logger.info(f"After coordinate filtering: {len(df)} KSI records")

    # Convert to GeoDataFrame
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
        crs=f"EPSG:{CRS_EPSG}",
    )

    # Clean date/time columns
    date_col = KSI_COLUMNS["date"]
    time_col = KSI_COLUMNS["time"]

    if date_col in gdf.columns:
        gdf["DATE"] = pd.to_datetime(gdf[date_col], errors="coerce")
        gdf = gdf.dropna(subset=["DATE"])

    if time_col in gdf.columns:
        # For KSI data, TIME is in minutes since midnight
        time_values = pd.to_numeric(gdf[time_col], errors="coerce")
        gdf["HOUR"] = (time_values // 60).astype(int)  # Convert minutes to hours
        gdf = gdf.dropna(subset=["HOUR"])

    logger.info(f"Final KSI data: {len(gdf)} records")
    return gdf


def load_road_network(data_dir: Path) -> gpd.GeoDataFrame:
    """
    Load road network geometry from GeoJSON

    Args:
        data_dir: Path to data directory

    Returns:
        Road network GeoDataFrame
    """
    logger.info("Loading road network...")

    file_path = data_dir / ROAD_NETWORK_FILE
    if not file_path.exists():
        raise FileNotFoundError(f"Road network file not found: {file_path}")

    # Load GeoJSON
    gdf = gpd.read_file(file_path)
    logger.info(f"Loaded {len(gdf)} road segments")

    # Ensure correct CRS
    if gdf.crs is None:
        gdf.set_crs(f"EPSG:{CRS_EPSG}", inplace=True)
    elif gdf.crs != f"EPSG:{CRS_EPSG}":
        gdf = gdf.to_crs(f"EPSG:{CRS_EPSG}")

    # Basic cleaning - remove segments with invalid geometry
    initial_count = len(gdf)
    gdf = gdf[gdf.geometry.is_valid]
    logger.info(f"Removed {initial_count - len(gdf)} segments with invalid geometry")

    # Road class: prefer FEATURE_CODE_DESC (Local, Major Arterial, Collector, etc.)
    # over LINEAR_NAME_TYPE (St, Blvd, Ave) for crash-relevant hierarchy
    if "FEATURE_CODE_DESC" in gdf.columns:
        gdf["ROAD_CLASS"] = gdf["FEATURE_CODE_DESC"].fillna("unknown").astype(str)
    elif ROAD_COLUMNS["road_class"] in gdf.columns:
        gdf["ROAD_CLASS"] = (
            gdf[ROAD_COLUMNS["road_class"]].fillna("unknown").astype(str)
        )
    else:
        gdf["ROAD_CLASS"] = "unknown"

    # One-way: binary indicator (NB, SB, EB, WB, ONE_WAY = one-way)
    if "ONEWAY_DIR_CODE" in gdf.columns:
        oneway_codes = {"NB", "SB", "EB", "WB", "ONE_WAY"}
        gdf["is_oneway"] = (
            gdf["ONEWAY_DIR_CODE"]
            .fillna("")
            .astype(str)
            .str.upper()
            .str.strip()
            .isin(oneway_codes)
        ).astype(int)
    else:
        gdf["is_oneway"] = 0

    # Intersection degree: how many segments meet at each intersection (graph degree)
    if "FROM_INTERSECTION_ID" in gdf.columns and "TO_INTERSECTION_ID" in gdf.columns:
        from_count = gdf.groupby("FROM_INTERSECTION_ID").size()
        to_count = gdf.groupby("TO_INTERSECTION_ID").size()
        all_ids = from_count.index.union(to_count.index).unique()
        degree = (
            from_count.reindex(all_ids).fillna(0) + to_count.reindex(all_ids).fillna(0)
        ).astype(int)
        gdf = gdf.merge(
            degree.rename("from_intersection_degree"),
            left_on="FROM_INTERSECTION_ID",
            right_index=True,
            how="left",
        )
        gdf = gdf.merge(
            degree.rename("to_intersection_degree"),
            left_on="TO_INTERSECTION_ID",
            right_index=True,
            how="left",
        )
        gdf["from_intersection_degree"] = (
            gdf["from_intersection_degree"].fillna(0).astype(int)
        )
        gdf["to_intersection_degree"] = (
            gdf["to_intersection_degree"].fillna(0).astype(int)
        )
    else:
        gdf["from_intersection_degree"] = 0
        gdf["to_intersection_degree"] = 0

    # Add segment length (convert to meters for filtering)
    # Convert to projected CRS for accurate length calculation
    gdf_projected = gdf.to_crs("EPSG:32617")  # UTM Zone 17N for Toronto
    gdf["segment_length"] = gdf_projected.geometry.length

    # Filter out very short segments (likely data artifacts)
    gdf = gdf[gdf["segment_length"] > 1]  # 1 meter minimum
    logger.info(f"After length filtering: {len(gdf)} road segments")

    logger.info(f"Final road network: {len(gdf)} segments")
    return gdf


def load_model_dataset(data_dir: Path) -> Optional[pd.DataFrame]:
    """
    Load optional traffic volume/speed data (model_dataset.csv) by segment.

    Keeps only traffic-related columns that are safe to use as features.
    Explicitly excludes crash_count / crash_rate to prevent target leakage.

    Returns None if file not found.
    """
    from config import MODEL_DATASET_FILE

    file_path = data_dir / MODEL_DATASET_FILE
    if not file_path.exists():
        logger.info(
            "Model dataset (%s) not found; proceeding without ADT/speed.", file_path
        )
        return None

    logger.info("Loading model dataset (ADT/speed) from %s", file_path)
    df = pd.read_csv(file_path)
    if "centreline_id" in df.columns:
        df = df.rename(columns={"centreline_id": "segment_id"})
    elif "CENTRELINE_ID" in df.columns:
        df = df.rename(columns={"CENTRELINE_ID": "segment_id"})
    else:
        logger.warning(
            "Model dataset has no centreline_id/CENTRELINE_ID column; skipping."
        )
        return None

    # Pick only traffic volume/speed columns — NOT crash-derived columns.
    # segment_length is excluded because the road network already has it
    # computed from geometry (more accurate than the CSV approximation).
    SAFE_COLS = [
        "avg_daily_vol",
        "avg_speed",
        "avg_85th_percentile_speed",
        "avg_95th_percentile_speed",
        "speed_variance",
        "exposure",
        "avg_wkdy_am_peak_vol",
        "avg_wkdy_pm_peak_vol",
        "avg_heavy_pct",
        "log_volume",
    ]
    keep = ["segment_id"] + [c for c in SAFE_COLS if c in df.columns]
    df = df[keep].drop_duplicates("segment_id")

    n_with_vol = (df["avg_daily_vol"] > 0).sum() if "avg_daily_vol" in df.columns else 0
    logger.info(
        "Loaded model dataset: %d segments, %d with non-zero avg_daily_vol. Columns: %s",
        len(df), n_with_vol, [c for c in keep if c != "segment_id"],
    )
    return df


def load_historical_weather(data_dir: Path) -> Optional[pd.DataFrame]:
    """
    Load historical weather from historicalweather.csv (NOAA GHCN daily data).
    Expands to hourly, converts units, and produces city-wide weather for panel join.

    Handles the standard NCEI/NOAA CSV export with columns:
    STATION, NAME, DATE, TAVG, TMAX, TMIN, PRCP, SNWD, AWND, etc.

    Output columns (per hourly row):
        datetime_hour, temperature (°C), precipitation (mm), snow_depth_mm,
        is_freezing (bool), is_precip (bool)

    No lat_grid/lon_grid — treated as city-wide; panel join uses datetime_hour only.
    Returns None if file not found.
    """
    try:
        from config import HISTORICAL_WEATHER_FILE
    except ImportError:
        return None

    file_path = data_dir / HISTORICAL_WEATHER_FILE
    if not file_path.exists():
        logger.info("Historical weather (%s) not found; proceeding without.", file_path)
        return None

    logger.info("Loading historical weather from %s", file_path)
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    logger.info("Raw weather CSV: %d rows, columns: %s", len(df), list(df.columns))

    # --- Parse date ---
    date_col = next(
        (c for c in df.columns if c.upper() == "DATE"),
        df.columns[0],
    )
    df["date"] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=["date"])

    # --- Helper: find column by substring (case-insensitive) ---
    def _find_col(pattern: str) -> Optional[str]:
        for c in df.columns:
            if c.upper() == pattern.upper():
                return c
        return None

    # --- Temperature: TAVG preferred, else (TMAX+TMIN)/2.  °F → °C ---
    tavg_col = _find_col("TAVG")
    tmax_col = _find_col("TMAX")
    tmin_col = _find_col("TMIN")

    if tavg_col:
        temp_f = pd.to_numeric(df[tavg_col], errors="coerce")
    elif tmax_col and tmin_col:
        tmax = pd.to_numeric(df[tmax_col], errors="coerce")
        tmin = pd.to_numeric(df[tmin_col], errors="coerce")
        temp_f = (tmax + tmin) / 2
    else:
        temp_f = pd.Series(np.nan, index=df.index)
    df["temperature"] = (temp_f - 32) * 5 / 9

    # --- Precipitation: inches → mm ---
    prcp_col = _find_col("PRCP")
    if prcp_col:
        df["precipitation"] = pd.to_numeric(df[prcp_col], errors="coerce").fillna(0) * 25.4
    else:
        df["precipitation"] = 0.0

    # --- Snow depth: SNWD (inches → mm).  NOAA uses SNWD for depth, SNOW for fall ---
    snwd_col = _find_col("SNWD") or _find_col("SNOW")
    if snwd_col:
        df["snow_depth_mm"] = pd.to_numeric(df[snwd_col], errors="coerce").fillna(0) * 25.4
    else:
        df["snow_depth_mm"] = 0.0

    # --- Wind speed: AWND (mph → m/s) if available ---
    awnd_col = _find_col("AWND")
    if awnd_col:
        df["wind_speed"] = pd.to_numeric(df[awnd_col], errors="coerce").fillna(0) * 0.44704
    else:
        df["wind_speed"] = 0.0

    # --- Derived binary flags ---
    df["is_freezing"] = (df["temperature"] <= 0).astype(int)
    df["is_precip"] = (df["precipitation"] > 0).astype(int)

    # --- Expand daily → hourly (one row per hour 00:00–23:00) ---
    feature_cols = ["temperature", "precipitation", "snow_depth_mm",
                    "wind_speed", "is_freezing", "is_precip"]
    dfs = []
    for h in range(24):
        w = df[["date"] + feature_cols].copy()
        w["datetime_hour"] = w["date"].dt.normalize() + pd.Timedelta(hours=h)
        w = w.drop(columns=["date"])
        dfs.append(w)
    weather = pd.concat(dfs, ignore_index=True)
    weather = weather.drop_duplicates(subset=["datetime_hour"]).sort_values("datetime_hour")

    logger.info(
        "Historical weather: %d hourly rows (%s to %s), features: %s",
        len(weather),
        weather["datetime_hour"].min(),
        weather["datetime_hour"].max(),
        feature_cols,
    )
    return weather


def merge_model_dataset_into_road_network(
    road_network: gpd.GeoDataFrame, model_dataset: Optional[pd.DataFrame]
) -> gpd.GeoDataFrame:
    """Left-join model_dataset (ADT, speed) onto road_network by segment_id."""
    if model_dataset is None or model_dataset.empty:
        return road_network

    road = road_network.copy()
    merge_cols = [c for c in model_dataset.columns if c != "segment_id"]
    road = road.merge(
        model_dataset[["segment_id"] + merge_cols],
        on="segment_id",
        how="left",
    )
    for c in merge_cols:
        road[c] = road[c].fillna(0)
    n_with_vol = (
        (road["avg_daily_vol"] > 0).sum() if "avg_daily_vol" in road.columns else 0
    )
    logger.info("Merged ADT/speed: %d segments with non-null avg_daily_vol", n_with_vol)
    return road


def load_and_clean_data(data_dir: Path):
    """
    Load and clean all three datasets

    Args:
        data_dir: Path to data directory

    Returns:
        Tuple of (collision_data, ksi_data, road_network)
    """
    logger.info("Starting data loading and cleaning process...")

    try:
        # Load all datasets
        collision_data = load_collision_data(data_dir)
        ksi_data = load_ksi_data(data_dir)
        road_network = load_road_network(data_dir)

        logger.info("Data loading completed successfully!")
        logger.info(f"Summary:")
        logger.info(f"  - Collision records: {len(collision_data)}")
        logger.info(f"  - KSI records: {len(ksi_data)}")
        logger.info(f"  - Road segments: {len(road_network)}")

        return collision_data, ksi_data, road_network

    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise


_TMC_DECADE_FILES = [
    "tmc_raw_data_2020_2029.csv",
    "tmc_raw_data_2010_2019.csv",
]
_SCHOOL_LOCATIONS_FILE = "School locations-all types data - 4326.csv"
_TTC_GTFS_DIR = "TTC Routes and Schedules Data"


def load_tmc_data(data_dir: Path, min_year: int = 2015) -> Optional[pd.DataFrame]:
    """
    Load and aggregate TMC intersection volume data from decade CSV files.

    Reads pedestrian, cyclist, and vehicle counts from 15-minute interval rows.
    Aggregates to daily totals per intersection then averages across all count dates.

    Returns a DataFrame with one row per intersection:
        centreline_id, longitude, latitude,
        tmc_daily_ped_vol, tmc_daily_cyclist_vol, tmc_daily_vehicle_vol

    Returns None if no TMC files are found.
    """
    dfs = []
    for fname in _TMC_DECADE_FILES:
        fpath = data_dir / fname
        if fpath.exists():
            df = pd.read_csv(fpath, low_memory=False)
            dfs.append(df)
            logger.info("Loaded TMC file: %s (%d rows)", fname, len(df))

    if not dfs:
        logger.info("No TMC files found; skipping TMC integration.")
        return None

    tmc = pd.concat(dfs, ignore_index=True)

    tmc["count_date"] = pd.to_datetime(tmc["count_date"], errors="coerce")
    tmc = tmc[tmc["count_date"].dt.year >= min_year].copy()
    logger.info("TMC records after year filter (>=%d): %d", min_year, len(tmc))

    ped_cols = ["n_appr_peds", "s_appr_peds", "e_appr_peds", "w_appr_peds"]
    bike_cols = ["n_appr_bike", "s_appr_bike", "e_appr_bike", "w_appr_bike"]
    vehicle_cols = [
        c for c in tmc.columns
        if any(x in c for x in ("appr_cars", "appr_truck", "appr_bus"))
    ]

    tmc["_total_peds"] = tmc[[c for c in ped_cols if c in tmc.columns]].sum(axis=1)
    tmc["_total_bikes"] = tmc[[c for c in bike_cols if c in tmc.columns]].sum(axis=1)
    tmc["_total_vehicles"] = tmc[[c for c in vehicle_cols if c in tmc.columns]].sum(axis=1)

    daily = tmc.groupby(["centreline_id", "count_date"]).agg(
        daily_ped_vol=("_total_peds", "sum"),
        daily_cyclist_vol=("_total_bikes", "sum"),
        daily_vehicle_vol=("_total_vehicles", "sum"),
        longitude=("longitude", "first"),
        latitude=("latitude", "first"),
    ).reset_index()

    result = daily.groupby("centreline_id").agg(
        tmc_daily_ped_vol=("daily_ped_vol", "mean"),
        tmc_daily_cyclist_vol=("daily_cyclist_vol", "mean"),
        tmc_daily_vehicle_vol=("daily_vehicle_vol", "mean"),
        longitude=("longitude", "first"),
        latitude=("latitude", "first"),
    ).reset_index()

    logger.info("TMC aggregated: %d unique intersections.", len(result))
    return result


def merge_tmc_into_road_network(
    road_network: gpd.GeoDataFrame,
    tmc_data: Optional[pd.DataFrame],
) -> gpd.GeoDataFrame:
    """
    Spatial join TMC intersection volumes to road segments.

    Each road segment gets the TMC volumes from intersections within 50m.
    If multiple intersections match, takes the max (worst-case exposure).
    Fills 0 for segments with no nearby intersection.
    """
    tmc_cols = ["tmc_daily_ped_vol", "tmc_daily_cyclist_vol", "tmc_daily_vehicle_vol"]
    if tmc_data is None or len(tmc_data) == 0:
        for col in tmc_cols:
            road_network[col] = 0.0
        return road_network

    tmc_gdf = gpd.GeoDataFrame(
        tmc_data,
        geometry=gpd.points_from_xy(tmc_data["longitude"], tmc_data["latitude"]),
        crs="EPSG:4326",
    ).to_crs("EPSG:32617")
    tmc_gdf["geometry"] = tmc_gdf.geometry.buffer(50)

    road_utm = road_network.to_crs("EPSG:32617").copy()
    road_centroids = gpd.GeoDataFrame(
        road_utm[["segment_id"]],
        geometry=road_utm.geometry.centroid,
        crs="EPSG:32617",
    )

    joined = gpd.sjoin(
        road_centroids,
        tmc_gdf[tmc_cols + ["geometry"]],
        how="left",
        predicate="within",
    )
    agg = joined.groupby("segment_id").agg(
        tmc_daily_ped_vol=("tmc_daily_ped_vol", "max"),
        tmc_daily_cyclist_vol=("tmc_daily_cyclist_vol", "max"),
        tmc_daily_vehicle_vol=("tmc_daily_vehicle_vol", "max"),
    ).reset_index()

    road_network = road_network.merge(agg, on="segment_id", how="left")
    for col in tmc_cols:
        road_network[col] = road_network[col].fillna(0.0)

    n_covered = (road_network["tmc_daily_ped_vol"] > 0).sum()
    logger.info(
        "TMC join: %d/%d segments have non-zero pedestrian volume.",
        n_covered, len(road_network),
    )
    return road_network


def load_school_locations(data_dir: Path) -> Optional[gpd.GeoDataFrame]:
    """
    Load school point locations from CSV with JSON geometry column.

    Parses the MultiPoint geometry JSON string and returns centroids as
    a GeoDataFrame in EPSG:4326. Returns None if file not found.
    """
    import json
    from shapely.geometry import shape

    fpath = data_dir / _SCHOOL_LOCATIONS_FILE
    if not fpath.exists():
        logger.info("School locations file (%s) not found; skipping.", fpath)
        return None

    df = pd.read_csv(fpath)
    logger.info("Loaded %d school records.", len(df))

    def _parse_geom(g: str):
        try:
            return shape(json.loads(g)).centroid
        except Exception:
            return None

    df["geometry"] = df["geometry"].apply(_parse_geom)
    df = df.dropna(subset=["geometry"])
    return gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")


def merge_school_zones_into_road_network(
    road_network: gpd.GeoDataFrame,
    schools: Optional[gpd.GeoDataFrame],
    buffer_m: int = 200,
) -> gpd.GeoDataFrame:
    """
    Add is_school_zone binary flag to road segments.

    A segment is marked 1 if its centroid falls within buffer_m meters of any school.
    """
    if schools is None or len(schools) == 0:
        road_network["is_school_zone"] = 0
        return road_network

    schools_utm = schools.to_crs("EPSG:32617").copy()
    schools_utm["geometry"] = schools_utm.geometry.buffer(buffer_m)

    road_utm = road_network.to_crs("EPSG:32617").copy()
    road_centroids = gpd.GeoDataFrame(
        road_utm[["segment_id"]],
        geometry=road_utm.geometry.centroid,
        crs="EPSG:32617",
    )

    joined = gpd.sjoin(
        road_centroids,
        schools_utm[["geometry"]],
        how="left",
        predicate="within",
    )
    school_segment_ids = set(joined.dropna(subset=["index_right"])["segment_id"].unique())
    road_network["is_school_zone"] = (
        road_network["segment_id"].isin(school_segment_ids).astype(int)
    )

    n_school_segs = int(road_network["is_school_zone"].sum())
    logger.info(
        "School zones: %d/%d segments within %dm of a school.",
        n_school_segs, len(road_network), buffer_m,
    )
    return road_network


def load_ttc_gtfs(data_dir: Path) -> Optional[pd.DataFrame]:
    """
    Load TTC GTFS and compute average trips per hour per stop.

    Reads stops.txt and stop_times.txt. GTFS arrival_time may exceed 24:00
    for overnight service — handles via modulo.

    Returns a DataFrame with: stop_id, stop_lat, stop_lon, avg_trips_per_hour
    Returns None if GTFS directory not found.
    """
    gtfs_dir = data_dir / _TTC_GTFS_DIR
    stops_path = gtfs_dir / "stops.txt"
    stop_times_path = gtfs_dir / "stop_times.txt"

    if not stops_path.exists() or not stop_times_path.exists():
        logger.info("TTC GTFS files not found at %s; skipping.", gtfs_dir)
        return None

    stops = pd.read_csv(stops_path)
    stop_times = pd.read_csv(stop_times_path, usecols=["stop_id", "arrival_time"])
    logger.info("GTFS: %d stops, %d stop_time rows.", len(stops), len(stop_times))

    def _parse_hour(t: str) -> Optional[int]:
        try:
            return int(str(t).split(":")[0]) % 24
        except Exception:
            return None

    stop_times["hour_of_day"] = stop_times["arrival_time"].apply(_parse_hour)
    stop_times = stop_times.dropna(subset=["hour_of_day"])

    trips_by_hour = (
        stop_times.groupby(["stop_id", "hour_of_day"])
        .size()
        .reset_index(name="trip_count")
    )
    avg_trips = (
        trips_by_hour.groupby("stop_id")["trip_count"]
        .mean()
        .reset_index(name="avg_trips_per_hour")
    )

    result = stops[["stop_id", "stop_lat", "stop_lon"]].merge(
        avg_trips, on="stop_id", how="left"
    )
    result["avg_trips_per_hour"] = result["avg_trips_per_hour"].fillna(0.0)
    logger.info("TTC GTFS: %d stops with frequency computed.", len(result))
    return result


def merge_ttc_into_road_network(
    road_network: gpd.GeoDataFrame,
    ttc_stops: Optional[pd.DataFrame],
    buffer_m: int = 150,
) -> gpd.GeoDataFrame:
    """
    Add nearby_transit_frequency to road segments.

    Each segment gets the sum of avg_trips_per_hour for all TTC stops within
    buffer_m meters. Represents pedestrian-generating transit activity.
    """
    if ttc_stops is None or len(ttc_stops) == 0:
        road_network["nearby_transit_frequency"] = 0.0
        return road_network

    ttc_gdf = gpd.GeoDataFrame(
        ttc_stops,
        geometry=gpd.points_from_xy(ttc_stops["stop_lon"], ttc_stops["stop_lat"]),
        crs="EPSG:4326",
    ).to_crs("EPSG:32617")
    ttc_gdf["geometry"] = ttc_gdf.geometry.buffer(buffer_m)

    road_utm = road_network.to_crs("EPSG:32617").copy()
    road_centroids = gpd.GeoDataFrame(
        road_utm[["segment_id"]],
        geometry=road_utm.geometry.centroid,
        crs="EPSG:32617",
    )

    joined = gpd.sjoin(
        road_centroids,
        ttc_gdf[["avg_trips_per_hour", "geometry"]],
        how="left",
        predicate="within",
    )
    agg = (
        joined.groupby("segment_id")["avg_trips_per_hour"]
        .sum()
        .reset_index(name="nearby_transit_frequency")
    )

    road_network = road_network.merge(agg, on="segment_id", how="left")
    road_network["nearby_transit_frequency"] = (
        road_network["nearby_transit_frequency"].fillna(0.0)
    )

    n_covered = (road_network["nearby_transit_frequency"] > 0).sum()
    logger.info(
        "TTC join: %d/%d segments within %dm of a TTC stop.",
        n_covered, len(road_network), buffer_m,
    )
    return road_network


if __name__ == "__main__":
    # Test the data loading
    logging.basicConfig(level=logging.INFO)
    data_dir = Path("data")
    collision_data, ksi_data, road_network = load_and_clean_data(data_dir)

    print(f"\nData Summary:")
    print(f"Collision data shape: {collision_data.shape}")
    print(f"KSI data shape: {ksi_data.shape}")
    print(f"Road network shape: {road_network.shape}")
