"""
Simplified spatial join module for Toronto Road Segment Crash Risk Prediction

This module performs spatial joins between crash points and road segments
using a simpler approach to avoid buffer-related issues.
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

def perform_spatial_join_simple(collision_data: gpd.GeoDataFrame, 
                               ksi_data: gpd.GeoDataFrame, 
                               road_network: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Perform spatial join between crash points and road segments using a simple approach
    
    Args:
        collision_data: GeoDataFrame of collision points
        ksi_data: GeoDataFrame of KSI points
        road_network: GeoDataFrame of road segments
        
    Returns:
        GeoDataFrame with crash data joined to road segments
    """
    logger.info("Starting simplified spatial join process...")
    
    # Create a copy of road network to avoid modifying original
    road_segments = road_network.copy()
    
    # Add unique segment ID if not present
    if 'segment_id' not in road_segments.columns:
        road_segments['segment_id'] = range(len(road_segments))
    
    # Initialize crash counts
    road_segments['num_total_crashes'] = 0
    road_segments['num_ksi_crashes'] = 0
    road_segments['fatality_count'] = 0
    
    logger.info(f"Processing {len(collision_data)} collision points...")
    
    # Process collision data using nearest neighbor approach
    collision_counts = _count_crashes_near_segments(collision_data, road_segments, 'collision')
    
    logger.info(f"Processing {len(ksi_data)} KSI points...")
    
    # Process KSI data
    ksi_counts = _count_crashes_near_segments(ksi_data, road_segments, 'ksi')
    
    # Update road segments with crash counts
    for segment_id in collision_counts:
        segment_idx = road_segments[road_segments['segment_id'] == segment_id].index[0]
        road_segments.loc[segment_idx, 'num_total_crashes'] = collision_counts[segment_id]['count']
        road_segments.loc[segment_idx, 'fatality_count'] += collision_counts[segment_id]['fatalities']
    
    for segment_id in ksi_counts:
        segment_idx = road_segments[road_segments['segment_id'] == segment_id].index[0]
        road_segments.loc[segment_idx, 'num_ksi_crashes'] = ksi_counts[segment_id]['count']
        road_segments.loc[segment_idx, 'fatality_count'] += ksi_counts[segment_id]['fatalities']
    
    logger.info("Simplified spatial join completed successfully!")
    logger.info(f"Road segments with crashes: {len(road_segments[road_segments['num_total_crashes'] > 0])}")
    
    return road_segments

def _count_crashes_near_segments(points_gdf: gpd.GeoDataFrame, 
                                segments_gdf: gpd.GeoDataFrame, 
                                crash_type: str) -> dict:
    """
    Count crashes near road segments using nearest neighbor approach
    
    Args:
        points_gdf: GeoDataFrame of crash points
        segments_gdf: GeoDataFrame of road segments
        crash_type: Type of crash ('collision' or 'ksi')
        
    Returns:
        Dictionary mapping segment IDs to crash counts
    """
    counts = {}
    
    # Convert to projected CRS for distance calculations
    points_proj = points_gdf.to_crs('EPSG:32617')  # UTM Zone 17N for Toronto
    segments_proj = segments_gdf.to_crs('EPSG:32617')
    
    logger.info(f"Finding nearest segments for {len(points_proj)} {crash_type} points...")
    
    # For each crash point, find the nearest road segment
    for idx, point in points_proj.iterrows():
        # Calculate distances to all segments
        distances = segments_proj.geometry.distance(point.geometry)
        nearest_segment_idx = distances.idxmin()
        nearest_distance = distances.min()
        
        # Only count if within reasonable distance (20 meters)
        if nearest_distance <= SPATIAL_BUFFER_DISTANCE:
            segment_id = segments_gdf.loc[nearest_segment_idx, 'segment_id']
            
            if segment_id not in counts:
                counts[segment_id] = {'count': 0, 'fatalities': 0}
            
            counts[segment_id]['count'] += 1
            
            # Count fatalities if available
            if crash_type == 'collision':
                injury_col = COLLISION_COLUMNS.get('injury', 'INJURY')
                fatalities_col = COLLISION_COLUMNS.get('fatalities', 'FATALITIES')
            else:  # ksi
                injury_col = KSI_COLUMNS.get('injury', 'INJURY')
                fatalities_col = KSI_COLUMNS.get('fatalities', 'FATAL_NO')
            
            if injury_col in point.index and pd.notna(point[injury_col]):
                if 'Fatal' in str(point[injury_col]).lower():
                    counts[segment_id]['fatalities'] += 1
            elif fatalities_col in point.index and pd.notna(point[fatalities_col]):
                counts[segment_id]['fatalities'] += int(point[fatalities_col])
    
    logger.info(f"Found {len(counts)} segments with {crash_type} crashes")
    return counts

def test_spatial_join_simple():
    """
    Test function for simplified spatial join
    """
    from data_loader import load_and_clean_data
    
    logging.basicConfig(level=logging.INFO)
    data_dir = Path("data")
    
    # Load data
    collision_data, ksi_data, road_network = load_and_clean_data(data_dir)
    
    # Perform spatial join
    segment_crashes = perform_spatial_join_simple(collision_data, ksi_data, road_network)
    
    # Print summary
    print(f"\nSimplified Spatial Join Results:")
    print(f"Total road segments: {len(segment_crashes)}")
    print(f"Segments with crashes: {len(segment_crashes[segment_crashes['num_total_crashes'] > 0])}")
    print(f"Segments with KSI: {len(segment_crashes[segment_crashes['num_ksi_crashes'] > 0])}")
    print(f"Total crashes: {segment_crashes['num_total_crashes'].sum()}")
    print(f"Total KSI: {segment_crashes['num_ksi_crashes'].sum()}")
    
    return segment_crashes

if __name__ == "__main__":
    test_spatial_join_simple() 