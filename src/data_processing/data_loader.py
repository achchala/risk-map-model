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


if __name__ == "__main__":
    # Test the data loading
    logging.basicConfig(level=logging.INFO)
    data_dir = Path("data")
    collision_data, ksi_data, road_network = load_and_clean_data(data_dir)

    print(f"\nData Summary:")
    print(f"Collision data shape: {collision_data.shape}")
    print(f"KSI data shape: {ksi_data.shape}")
    print(f"Road network shape: {road_network.shape}")
