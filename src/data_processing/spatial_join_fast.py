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
    logger.info("Starting fast spatial join process...")
    
    # Create a copy of road network to avoid modifying original
    road_segments = road_network.copy()
    
    # Add unique segment ID if not present
    if 'segment_id' not in road_segments.columns:
        road_segments['segment_id'] = range(len(road_segments))
    
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
        
        if injury_col in point.index and pd.notna(point[injury_col]):
            if 'Fatal' in str(point[injury_col]).lower():
                counts[segment_id]['fatalities'] += 1
        elif fatalities_col in point.index and pd.notna(point[fatalities_col]):
            counts[segment_id]['fatalities'] += int(point[fatalities_col])
    
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