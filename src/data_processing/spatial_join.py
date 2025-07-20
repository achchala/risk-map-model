"""
Spatial join module for Toronto Road Segment Crash Risk Prediction

This module performs spatial joins between crash points and road segments
using a buffer-based approach to handle GPS errors and coordinate snapping.
"""

import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path
import logging
from shapely.geometry import Point
import sys

# Add parent directory to path for config import
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import *

logger = logging.getLogger(__name__)

def perform_spatial_join(collision_data: gpd.GeoDataFrame, 
                        ksi_data: gpd.GeoDataFrame, 
                        road_network: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Perform spatial join between crash points and road segments
    
    Args:
        collision_data: GeoDataFrame of collision points
        ksi_data: GeoDataFrame of KSI points
        road_network: GeoDataFrame of road segments
        
    Returns:
        GeoDataFrame with crash data joined to road segments
    """
    logger.info("Starting spatial join process...")
    
    # Create a copy of road network to avoid modifying original
    road_segments = road_network.copy()
    
    # Add unique segment ID if not present
    if 'segment_id' not in road_segments.columns:
        road_segments['segment_id'] = range(len(road_segments))
    
    # Initialize crash counts
    road_segments['num_total_crashes'] = 0
    road_segments['num_ksi_crashes'] = 0
    road_segments['crash_points'] = []
    road_segments['ksi_points'] = []
    
    logger.info(f"Processing {len(collision_data)} collision points...")
    
    # Process collision data
    collision_matches = _match_points_to_segments(
        collision_data, road_segments, 'collision'
    )
    
    logger.info(f"Processing {len(ksi_data)} KSI points...")
    
    # Process KSI data
    ksi_matches = _match_points_to_segments(
        ksi_data, road_segments, 'ksi'
    )
    
    # Aggregate results
    road_segments = _aggregate_crash_data(
        road_segments, collision_matches, ksi_matches
    )
    
    logger.info("Spatial join completed successfully!")
    logger.info(f"Road segments with crashes: {len(road_segments[road_segments['num_total_crashes'] > 0])}")
    
    return road_segments

def _match_points_to_segments(points_gdf: gpd.GeoDataFrame, 
                            segments_gdf: gpd.GeoDataFrame, 
                            crash_type: str) -> dict:
    """
    Match crash points to road segments using buffer-based spatial join
    
    Args:
        points_gdf: GeoDataFrame of crash points
        segments_gdf: GeoDataFrame of road segments
        crash_type: Type of crash ('collision' or 'ksi')
        
    Returns:
        Dictionary mapping segment IDs to crash points
    """
    matches = {}
    
    # Create buffer around road segments for matching
    segments_buffered = segments_gdf.copy()
    # Convert to projected CRS for accurate buffer operation
    segments_projected = segments_buffered.to_crs('EPSG:32617')  # UTM Zone 17N for Toronto
    buffered_geometries = segments_projected.geometry.buffer(SPATIAL_BUFFER_DISTANCE)
    segments_buffered['geometry'] = buffered_geometries
    # Convert back to original CRS
    segments_buffered = segments_buffered.to_crs(segments_gdf.crs)
    
    # Perform spatial join
    logger.info(f"Performing spatial join with {len(points_gdf)} points and {len(segments_buffered)} segments...")
    joined = gpd.sjoin(points_gdf, segments_buffered, how='left', predicate='within')
    logger.info(f"Spatial join completed. Found {len(joined)} matches.")
    
    # Group by segment and collect crash points
    for segment_id in segments_gdf['segment_id']:
        segment_crashes = joined[joined['segment_id'] == segment_id]
        if len(segment_crashes) > 0:
            matches[segment_id] = segment_crashes
    
    logger.info(f"Matched {len(matches)} segments with {crash_type} crashes")
    return matches

def _aggregate_crash_data(road_segments: gpd.GeoDataFrame,
                         collision_matches: dict,
                         ksi_matches: dict) -> gpd.GeoDataFrame:
    """
    Aggregate crash data for each road segment
    
    Args:
        road_segments: GeoDataFrame of road segments
        collision_matches: Dictionary of collision matches
        ksi_matches: Dictionary of KSI matches
        
    Returns:
        Updated road segments with crash counts and data
    """
    logger.info("Aggregating crash data by road segment...")
    
    # Initialize crash data columns
    road_segments['num_total_crashes'] = 0
    road_segments['num_ksi_crashes'] = 0
    road_segments['collision_dates'] = []
    road_segments['collision_hours'] = []
    road_segments['ksi_dates'] = []
    road_segments['ksi_hours'] = []
    road_segments['fatality_count'] = 0
    
    # Process collision matches
    for segment_id, crashes in collision_matches.items():
        segment_idx = road_segments[road_segments['segment_id'] == segment_id].index[0]
        
        road_segments.loc[segment_idx, 'num_total_crashes'] = len(crashes)
        
        # Collect dates and hours
        if 'DATE' in crashes.columns:
            dates = crashes['DATE'].dropna().tolist()
            road_segments.loc[segment_idx, 'collision_dates'] = dates
        
        if 'HOUR' in crashes.columns:
            hours = crashes['HOUR'].dropna().tolist()
            road_segments.loc[segment_idx, 'collision_hours'] = hours
        
        # Count fatalities if available
        injury_col = COLLISION_COLUMNS.get('injury', 'INJURY')
        fatalities_col = COLLISION_COLUMNS.get('fatalities', 'FATALITIES')
        
        if injury_col in crashes.columns:
            fatality_count = len(crashes[crashes[injury_col].str.contains('Fatal', case=False, na=False)])
            road_segments.loc[segment_idx, 'fatality_count'] += fatality_count
        elif fatalities_col in crashes.columns:
            fatality_count = crashes[fatalities_col].sum()
            road_segments.loc[segment_idx, 'fatality_count'] += fatality_count
    
    # Process KSI matches
    for segment_id, crashes in ksi_matches.items():
        segment_idx = road_segments[road_segments['segment_id'] == segment_id].index[0]
        
        road_segments.loc[segment_idx, 'num_ksi_crashes'] = len(crashes)
        
        # Collect dates and hours
        if 'DATE' in crashes.columns:
            dates = crashes['DATE'].dropna().tolist()
            road_segments.loc[segment_idx, 'ksi_dates'] = dates
        
        if 'HOUR' in crashes.columns:
            hours = crashes['HOUR'].dropna().tolist()
            road_segments.loc[segment_idx, 'collision_hours'] = hours
        
        # Count fatalities if available
        injury_col = KSI_COLUMNS.get('injury', 'INJURY')
        fatalities_col = KSI_COLUMNS.get('fatalities', 'FATAL_NO')
        
        if injury_col in crashes.columns:
            fatality_count = len(crashes[crashes[injury_col].str.contains('Fatal', case=False, na=False)])
            road_segments.loc[segment_idx, 'fatality_count'] += fatality_count
        elif fatalities_col in crashes.columns:
            fatality_count = crashes[fatalities_col].sum()
            road_segments.loc[segment_idx, 'fatality_count'] += fatality_count
    
    # Convert lists to strings for storage
    road_segments['collision_dates'] = road_segments['collision_dates'].apply(
        lambda x: ','.join([str(d) for d in x]) if isinstance(x, list) else ''
    )
    road_segments['collision_hours'] = road_segments['collision_hours'].apply(
        lambda x: ','.join([str(h) for h in x]) if isinstance(x, list) else ''
    )
    road_segments['ksi_dates'] = road_segments['ksi_dates'].apply(
        lambda x: ','.join([str(d) for d in x]) if isinstance(x, list) else ''
    )
    road_segments['ksi_hours'] = road_segments['ksi_hours'].apply(
        lambda x: ','.join([str(h) for h in x]) if isinstance(x, list) else ''
    )
    
    logger.info(f"Aggregation complete. Segments with crashes: {len(road_segments[road_segments['num_total_crashes'] > 0])}")
    
    return road_segments

def test_spatial_join():
    """
    Test function for spatial join functionality
    """
    from data_loader import load_and_clean_data
    
    logging.basicConfig(level=logging.INFO)
    data_dir = Path("data")
    
    # Load data
    collision_data, ksi_data, road_network = load_and_clean_data(data_dir)
    
    # Perform spatial join
    segment_crashes = perform_spatial_join(collision_data, ksi_data, road_network)
    
    # Print summary
    print(f"\nSpatial Join Results:")
    print(f"Total road segments: {len(segment_crashes)}")
    print(f"Segments with crashes: {len(segment_crashes[segment_crashes['num_total_crashes'] > 0])}")
    print(f"Segments with KSI: {len(segment_crashes[segment_crashes['num_ksi_crashes'] > 0])}")
    print(f"Total crashes: {segment_crashes['num_total_crashes'].sum()}")
    print(f"Total KSI: {segment_crashes['num_ksi_crashes'].sum()}")
    
    return segment_crashes

if __name__ == "__main__":
    test_spatial_join() 