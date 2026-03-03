"""
Fast spatial join module for Toronto Road Segment Crash Risk Prediction

This module performs spatial joins between crash points and road segments
using efficient vectorized operations and batch processing.
"""

import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path
import logging
from shapely.geometry import Point
import sys
from sklearn.neighbors import BallTree
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for config import
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import *

logger = logging.getLogger(__name__)


def verify_crs_and_distance(road_network: gpd.GeoDataFrame, buffer_distance_m: float = SPATIAL_BUFFER_DISTANCE) -> bool:
    """
    Verify that the road network CRS and buffer distance are configured correctly.

    This helps catch subtle bugs where distance-based joins are performed in
    degrees instead of meters.
    """
    # 1. Check CRS is set
    if road_network.crs is None:
        raise ValueError("Road network CRS is not set")

    # 2. Convert to projected CRS (meters) for distance calculations
    road_proj = road_network.to_crs("EPSG:32617")  # UTM Zone 17N (Toronto)

    # 3. Verify buffer distance behaves like meters
    test_point = road_proj.geometry.iloc[0].centroid
    test_buffer = test_point.buffer(buffer_distance_m)
    buffer_area = test_buffer.area

    # Expected area for circular buffer: π · r²
    expected_area = np.pi * buffer_distance_m**2
    if abs(buffer_area - expected_area) > 100:  # ~100 m² tolerance
        raise ValueError(
            f"Buffer distance appears incorrect. Expected area ~{expected_area:.0f} m², "
            f"got {buffer_area:.0f} m²."
        )

    logger.info(
        "CRS verified: %s, buffer distance %.1fm confirmed",
        road_network.crs,
        buffer_distance_m,
    )
    return True


def _ensure_stable_segment_id(road_network: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Ensure a stable segment_id column exists, preferring CENTRELINE_ID.

    This is critical so that segment IDs remain consistent across:
    - event-level assignments
    - panel dataset construction
    - modeling
    - routing graph construction
    """
    road_segments = road_network.copy()

    if "CENTRELINE_ID" in road_segments.columns:
        road_segments["segment_id"] = road_segments["CENTRELINE_ID"]
    else:
        # Fall back to an index-based ID, but keep it stable and string-typed
        if "segment_id" not in road_segments.columns:
            road_segments["segment_id"] = road_segments.index.astype(str)

    return road_segments

def perform_spatial_join_fast(collision_data: gpd.GeoDataFrame, 
                             ksi_data: gpd.GeoDataFrame, 
                             road_network: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Perform spatial join between crash points and road segments using fast approach
    
    Args:
        collision_data: GeoDataFrame of collision points
        ksi_data: GeoDataFrame of KSI points
        road_network: GeoDataFrame of road segments
        
    Returns:
        GeoDataFrame with crash data joined to road segments
    """
    logger.info("Starting fast spatial join process (aggregate counts)...")
    
    # Create a copy of road network to avoid modifying original and
    # ensure we have a stable segment_id (prefer CENTRELINE_ID).
    road_segments = _ensure_stable_segment_id(road_network)
    verify_crs_and_distance(road_segments, SPATIAL_BUFFER_DISTANCE)
    
    # Initialize crash counts
    road_segments['num_total_crashes'] = 0
    road_segments['num_ksi_crashes'] = 0
    road_segments['fatality_count'] = 0
    
    # Convert to projected CRS for accurate distance calculations
    road_proj = road_segments.to_crs('EPSG:32617')  # UTM Zone 17N for Toronto
    
    # Get road segment centroids for distance calculations
    road_centroids = road_proj.geometry.centroid
    road_coords = np.array([[point.x, point.y] for point in road_centroids])
    
    logger.info(f"Processing {len(collision_data)} collision points...")
    
    # Process collision data
    collision_counts = _count_crashes_fast(collision_data, road_coords, road_segments, 'collision')
    
    logger.info(f"Processing {len(ksi_data)} KSI points...")
    
    # Process KSI data
    ksi_counts = _count_crashes_fast(ksi_data, road_coords, road_segments, 'ksi')
    
    # Update road segments with crash counts
    for segment_id in collision_counts:
        segment_idx = road_segments[road_segments['segment_id'] == segment_id].index[0]
        road_segments.loc[segment_idx, 'num_total_crashes'] = collision_counts[segment_id]['count']
        road_segments.loc[segment_idx, 'fatality_count'] += collision_counts[segment_id]['fatalities']
    
    for segment_id in ksi_counts:
        segment_idx = road_segments[road_segments['segment_id'] == segment_id].index[0]
        road_segments.loc[segment_idx, 'num_ksi_crashes'] = ksi_counts[segment_id]['count']
        road_segments.loc[segment_idx, 'fatality_count'] += ksi_counts[segment_id]['fatalities']
    
    logger.info("Fast spatial join completed successfully!")
    logger.info(f"Road segments with crashes: {len(road_segments[road_segments['num_total_crashes'] > 0])}")
    
    return road_segments


def _compute_event_fatalities(points_gdf: gpd.GeoDataFrame, crash_type: str) -> pd.Series:
    """
    Compute fatalities for each crash event in a vectorized way.

    Logic mirrors the per-point counting in _count_crashes_fast but returns
    a per-row integer Series instead of aggregated counts.
    """
    if crash_type == "collision":
        injury_col = COLLISION_COLUMNS.get("injury", "INJURY")
        fatalities_col = COLLISION_COLUMNS.get("fatalities", "FATALITIES")
    else:
        injury_col = KSI_COLUMNS.get("injury", "INJURY")
        fatalities_col = KSI_COLUMNS.get("fatalities", "FATAL_NO")

    fatalities = pd.Series(0, index=points_gdf.index, dtype="int64")

    # 1) Use explicit fatalities column where available
    if fatalities_col in points_gdf.columns:
        raw = points_gdf[fatalities_col]
        # Try numeric first
        numeric = pd.to_numeric(raw, errors="coerce")
        numeric = numeric.fillna(0)
        numeric[numeric < 0] = 0
        fatalities = numeric.astype("int64")

    # 2) If fatalities still zero, fall back to injury text containing "Fatal"
    if injury_col in points_gdf.columns:
        injury_str = points_gdf[injury_col].astype(str).str.lower()
        fatal_mask = injury_str.str.contains("fatal", na=False) & (fatalities == 0)
        fatalities.loc[fatal_mask] = 1

    return fatalities


def perform_spatial_join_event_level(
    collision_data: gpd.GeoDataFrame,
    ksi_data: gpd.GeoDataFrame,
    road_network: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Event-level spatial join between crash points and road segments.

    Output schema (one row per crash event that can be assigned to a segment):
        - segment_id (stable, based on CENTRELINE_ID where available)
        - event_datetime (timestamp combining date + time)
        - crash_type ('collision' or 'ksi')
        - is_ksi (boolean)
        - fatalities (integer count)
        - geometry (Point, original CRS)
    """
    logger.info("Starting event-level spatial join process...")

    # 1. Ensure stable segment_id and verify CRS / distance behaviour
    road_segments = _ensure_stable_segment_id(road_network)
    verify_crs_and_distance(road_segments, SPATIAL_BUFFER_DISTANCE)

    # 2. Build BallTree on road centroids in projected CRS
    road_proj = road_segments.to_crs("EPSG:32617")
    road_centroids = road_proj.geometry.centroid
    road_coords = np.array([[p.x, p.y] for p in road_centroids])
    tree = BallTree(road_coords, metric="euclidean")

    assignments = []

    # Helper to process one crash dataset
    def _assign_events(
        crashes: gpd.GeoDataFrame,
        crash_type: str,
    ) -> pd.DataFrame:
        if crashes is None or crashes.empty:
            return pd.DataFrame(
                columns=[
                    "segment_id",
                    "event_datetime",
                    "crash_type",
                    "is_ksi",
                    "fatalities",
                    "geometry",
                ]
            )

        logger.info("Processing %d %s events for event-level join...", len(crashes), crash_type)

        # 2a. Compute event_datetime (DATE + HOUR or DATE + TIME in minutes)
        df = crashes.copy()
        if crash_type == "collision":
            date_col = COLLISION_COLUMNS.get("date", "OCC_DATE")
            hour_col = COLLISION_COLUMNS.get("time", "OCC_HOUR")
            hours = pd.to_numeric(df[hour_col], errors="coerce")
            df["event_datetime"] = pd.to_datetime(df[date_col], errors="coerce") + pd.to_timedelta(
                hours, unit="h"
            )
            is_ksi_flag = False
        else:
            date_col = KSI_COLUMNS.get("date", "DATE")
            time_col = KSI_COLUMNS.get("time", "TIME")
            minutes = pd.to_numeric(df[time_col], errors="coerce")
            df["event_datetime"] = pd.to_datetime(df[date_col], errors="coerce") + pd.to_timedelta(
                minutes, unit="m"
            )
            is_ksi_flag = True

        # Drop any rows where we failed to construct event_datetime
        before = len(df)
        df = df.dropna(subset=["event_datetime"])
        logger.info(
            "After datetime cleaning: %d %s events (dropped %d)",
            len(df),
            crash_type,
            before - len(df),
        )

        if df.empty:
            return pd.DataFrame(
                columns=[
                    "segment_id",
                    "event_datetime",
                    "crash_type",
                    "is_ksi",
                    "fatalities",
                    "geometry",
                ]
            )

        # 2b. Project crash points for distance queries
        points_proj = df.to_crs("EPSG:32617")
        point_coords = np.array([[p.x, p.y] for p in points_proj.geometry])

        # 2c. Nearest segment for each crash
        distances, indices = tree.query(point_coords, k=1)
        distances = distances.flatten()
        indices = indices.flatten()

        within_buffer = distances <= SPATIAL_BUFFER_DISTANCE
        if not np.any(within_buffer):
            logger.info("No %s events found within %.1fm buffer of any segment.", crash_type, SPATIAL_BUFFER_DISTANCE)
            return pd.DataFrame(
                columns=[
                    "segment_id",
                    "event_datetime",
                    "crash_type",
                    "is_ksi",
                    "fatalities",
                    "geometry",
                ]
            )

        valid_indices = indices[within_buffer]
        valid_rows = df.iloc[np.where(within_buffer)[0]].copy()

        logger.info(
            "Assigned %d of %d %s events to segments within %.1fm.",
            len(valid_rows),
            len(df),
            crash_type,
            SPATIAL_BUFFER_DISTANCE,
        )

        # 2d. Compute fatalities per event
        fatalities = _compute_event_fatalities(valid_rows, crash_type=crash_type)

        # 2e. Build event-level assignment DataFrame (vectorized)
        segment_ids = road_segments.iloc[valid_indices]["segment_id"].values
        event_df = pd.DataFrame(
            {
                "segment_id": segment_ids,
                "event_datetime": valid_rows["event_datetime"].values,
                "crash_type": crash_type,
                "is_ksi": is_ksi_flag,
                "fatalities": fatalities.values,
                "geometry": valid_rows.geometry.values,
            }
        )

        return event_df

    # Process collision and KSI datasets
    collision_assignments = _assign_events(collision_data, "collision")
    ksi_assignments = _assign_events(ksi_data, "ksi")

    if not collision_assignments.empty:
        assignments.append(collision_assignments)
    if not ksi_assignments.empty:
        assignments.append(ksi_assignments)

    if not assignments:
        logger.warning("No event-level crash assignments produced.")
        return gpd.GeoDataFrame(
            columns=[
                "segment_id",
                "event_datetime",
                "crash_type",
                "is_ksi",
                "fatalities",
                "geometry",
            ],
            geometry="geometry",
            crs=road_network.crs,
        )

    all_events = pd.concat(assignments, ignore_index=True)
    event_gdf = gpd.GeoDataFrame(all_events, geometry="geometry", crs=road_network.crs)

    logger.info("Event-level spatial join completed with %d crash events.", len(event_gdf))
    return event_gdf

def _count_crashes_fast(points_gdf: gpd.GeoDataFrame, 
                       road_coords: np.ndarray,
                       road_segments: gpd.GeoDataFrame, 
                       crash_type: str) -> dict:
    """
    Count crashes near road segments using fast nearest neighbor approach
    
    Args:
        points_gdf: GeoDataFrame of crash points
        road_coords: Array of road segment centroid coordinates
        road_segments: GeoDataFrame of road segments
        crash_type: Type of crash ('collision' or 'ksi')
        
    Returns:
        Dictionary mapping segment IDs to crash counts
    """
    counts = {}
    
    # Convert to projected CRS for distance calculations
    points_proj = points_gdf.to_crs('EPSG:32617')  # UTM Zone 17N for Toronto
    
    # Get point coordinates
    point_coords = np.array([[point.x, point.y] for point in points_proj.geometry])
    
    logger.info(f"Finding nearest segments for {len(points_proj)} {crash_type} points...")
    
    # Use BallTree for efficient nearest neighbor search
    tree = BallTree(road_coords, metric='euclidean')
    
    # Find nearest segment for each point
    distances, indices = tree.query(point_coords, k=1)
    
    # Count crashes within buffer distance
    within_buffer = distances.flatten() <= SPATIAL_BUFFER_DISTANCE
    
    # Get segment IDs for points within buffer
    valid_indices = indices.flatten()[within_buffer]
    valid_points = points_proj.iloc[np.where(within_buffer)[0]]
    
    logger.info(f"Found {len(valid_points)} {crash_type} points within buffer distance")
    
    # Count crashes by segment
    for i, point_idx in enumerate(valid_points.index):
        segment_idx = valid_indices[i]
        segment_id = road_segments.iloc[segment_idx]['segment_id']
        
        if segment_id not in counts:
            counts[segment_id] = {'count': 0, 'fatalities': 0}
        
        counts[segment_id]['count'] += 1
        
        # Count fatalities if available
        point = valid_points.loc[point_idx]
        if crash_type == 'collision':
            injury_col = COLLISION_COLUMNS.get('injury', 'INJURY')
            fatalities_col = COLLISION_COLUMNS.get('fatalities', 'FATALITIES')
        else:  # ksi
            injury_col = KSI_COLUMNS.get('injury', 'INJURY')
            fatalities_col = KSI_COLUMNS.get('fatalities', 'FATAL_NO')
        
        # Prioritize fatalities column (actual count) over injury column (text description)
        fatalities_count = 0
        if fatalities_col in point.index and pd.notna(point[fatalities_col]):
            try:
                # Try to convert to numeric first (handles float64, int64, etc.)
                fatalities_value = pd.to_numeric(point[fatalities_col], errors='coerce')
                if pd.notna(fatalities_value) and fatalities_value > 0:
                    fatalities_count = int(fatalities_value)
            except (ValueError, TypeError):
                # If conversion fails, try direct int conversion
                try:
                    fatalities_count = int(point[fatalities_col])
                except (ValueError, TypeError):
                    fatalities_count = 0
        
        # If fatalities column doesn't have a count, check injury column for "Fatal" indicator
        # (but still use actual count if available)
        if fatalities_count == 0 and injury_col in point.index and pd.notna(point[injury_col]):
            if 'Fatal' in str(point[injury_col]).lower():
                fatalities_count = 1
        
        # Add the fatalities count
        if fatalities_count > 0:
            counts[segment_id]['fatalities'] += fatalities_count
    
    logger.info(f"Found {len(counts)} segments with {crash_type} crashes")
    return counts

def test_spatial_join_fast():
    """
    Test function for fast spatial join
    """
    from data_loader import load_and_clean_data
    
    logging.basicConfig(level=logging.INFO)
    data_dir = Path("data")
    
    # Load data
    collision_data, ksi_data, road_network = load_and_clean_data(data_dir)
    
    # Perform spatial join
    segment_crashes = perform_spatial_join_fast(collision_data, ksi_data, road_network)
    
    # Print summary
    print(f"\nFast Spatial Join Results:")
    print(f"Total road segments: {len(segment_crashes)}")
    print(f"Segments with crashes: {len(segment_crashes[segment_crashes['num_total_crashes'] > 0])}")
    print(f"Segments with KSI: {len(segment_crashes[segment_crashes['num_ksi_crashes'] > 0])}")
    print(f"Total crashes: {segment_crashes['num_total_crashes'].sum()}")
    print(f"Total KSI: {segment_crashes['num_ksi_crashes'].sum()}")
    
    return segment_crashes

if __name__ == "__main__":
    test_spatial_join_fast() 