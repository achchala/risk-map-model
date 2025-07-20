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
    lat_col = COLLISION_COLUMNS['latitude']
    lon_col = COLLISION_COLUMNS['longitude']
    
    initial_count = len(df)
    df = df.dropna(subset=[lat_col, lon_col])
    dropped_count = initial_count - len(df)
    logger.info(f"Dropped {dropped_count} records with missing coordinates")
    
    # Convert coordinates to numeric
    df[lat_col] = pd.to_numeric(df[lat_col], errors='coerce')
    df[lon_col] = pd.to_numeric(df[lon_col], errors='coerce')
    
    # Filter out invalid coordinates (outside reasonable bounds for Toronto)
    df = df[
        (df[lat_col] >= 43.5) & (df[lat_col] <= 44.0) &
        (df[lon_col] >= -79.8) & (df[lon_col] <= -79.0)
    ]
    logger.info(f"After coordinate filtering: {len(df)} records")
    
    # Convert to GeoDataFrame
    gdf = gpd.GeoDataFrame(
        df, 
        geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
        crs=f"EPSG:{CRS_EPSG}"
    )
    
    # Clean date/time columns
    date_col = COLLISION_COLUMNS['date']
    time_col = COLLISION_COLUMNS['time']
    
    if date_col in gdf.columns:
        gdf['DATE'] = pd.to_datetime(gdf[date_col], errors='coerce')
        gdf = gdf.dropna(subset=['DATE'])
    
    if time_col in gdf.columns:
        # Convert time to hour of day (already numeric for collision data)
        gdf['HOUR'] = pd.to_numeric(gdf[time_col], errors='coerce')
        gdf = gdf.dropna(subset=['HOUR'])
    
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
    lat_col = KSI_COLUMNS['latitude']
    lon_col = KSI_COLUMNS['longitude']
    
    initial_count = len(df)
    df = df.dropna(subset=[lat_col, lon_col])
    dropped_count = initial_count - len(df)
    logger.info(f"Dropped {dropped_count} KSI records with missing coordinates")
    
    # Convert coordinates to numeric
    df[lat_col] = pd.to_numeric(df[lat_col], errors='coerce')
    df[lon_col] = pd.to_numeric(df[lon_col], errors='coerce')
    
    # Filter out invalid coordinates
    df = df[
        (df[lat_col] >= 43.5) & (df[lat_col] <= 44.0) &
        (df[lon_col] >= -79.8) & (df[lon_col] <= -79.0)
    ]
    logger.info(f"After coordinate filtering: {len(df)} KSI records")
    
    # Convert to GeoDataFrame
    gdf = gpd.GeoDataFrame(
        df, 
        geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
        crs=f"EPSG:{CRS_EPSG}"
    )
    
    # Clean date/time columns
    date_col = KSI_COLUMNS['date']
    time_col = KSI_COLUMNS['time']
    
    if date_col in gdf.columns:
        gdf['DATE'] = pd.to_datetime(gdf[date_col], errors='coerce')
        gdf = gdf.dropna(subset=['DATE'])
    
    if time_col in gdf.columns:
        # For KSI data, TIME is in minutes since midnight
        time_values = pd.to_numeric(gdf[time_col], errors='coerce')
        gdf['HOUR'] = (time_values // 60).astype(int)  # Convert minutes to hours
        gdf = gdf.dropna(subset=['HOUR'])
    
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
    
    # Add road class column if it exists
    road_class_col = ROAD_COLUMNS['road_class']
    if road_class_col in gdf.columns:
        gdf['ROAD_CLASS'] = gdf[road_class_col]
    else:
        gdf['ROAD_CLASS'] = 'unknown'
    
    # Add segment length (convert to meters for filtering)
    # Convert to projected CRS for accurate length calculation
    gdf_projected = gdf.to_crs('EPSG:32617')  # UTM Zone 17N for Toronto
    gdf['segment_length'] = gdf_projected.geometry.length
    
    # Filter out very short segments (likely data artifacts)
    gdf = gdf[gdf['segment_length'] > 1]  # 1 meter minimum
    logger.info(f"After length filtering: {len(gdf)} road segments")
    
    logger.info(f"Final road network: {len(gdf)} segments")
    return gdf

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