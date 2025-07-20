#!/usr/bin/env python3
"""
Data exploration script to understand the structure of the datasets
"""

import pandas as pd
import geopandas as gpd
from pathlib import Path
import sys

# Add parent directory to path for config import
sys.path.append(str(Path(__file__).parent))
from config import *

def explore_collision_data():
    """Explore the collision data structure"""
    print("=" * 60)
    print("EXPLORING COLLISION DATA")
    print("=" * 60)
    
    file_path = Path("data") / COLLISION_DATA_FILE
    print(f"Loading: {file_path}")
    
    # Load Excel file
    df = pd.read_excel(file_path)
    
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"First few rows:")
    print(df.head())
    
    # Check for coordinate columns
    coord_cols = [col for col in df.columns if any(coord in col.upper() for coord in ['LAT', 'LONG', 'LON', 'X', 'Y', 'COORD'])]
    print(f"\nPotential coordinate columns: {coord_cols}")
    
    # Check for date/time columns
    date_cols = [col for col in df.columns if any(date in col.upper() for date in ['DATE', 'TIME', 'HOUR'])]
    print(f"Potential date/time columns: {date_cols}")
    
    # Check for injury columns
    injury_cols = [col for col in df.columns if any(injury in col.upper() for injury in ['INJURY', 'FATAL', 'SEVERITY'])]
    print(f"Potential injury columns: {injury_cols}")
    
    return df

def explore_ksi_data():
    """Explore the KSI data structure"""
    print("\n" + "=" * 60)
    print("EXPLORING KSI DATA")
    print("=" * 60)
    
    file_path = Path("data") / KSI_DATA_FILE
    print(f"Loading: {file_path}")
    
    # Load CSV file
    df = pd.read_csv(file_path)
    
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"First few rows:")
    print(df.head())
    
    # Check for coordinate columns
    coord_cols = [col for col in df.columns if any(coord in col.upper() for coord in ['LAT', 'LONG', 'LON', 'X', 'Y', 'COORD'])]
    print(f"\nPotential coordinate columns: {coord_cols}")
    
    # Check for date/time columns
    date_cols = [col for col in df.columns if any(date in col.upper() for date in ['DATE', 'TIME', 'HOUR'])]
    print(f"Potential date/time columns: {date_cols}")
    
    return df

def explore_road_network():
    """Explore the road network structure"""
    print("\n" + "=" * 60)
    print("EXPLORING ROAD NETWORK")
    print("=" * 60)
    
    file_path = Path("data") / ROAD_NETWORK_FILE
    print(f"Loading: {file_path}")
    
    # Load GeoJSON file
    gdf = gpd.read_file(file_path)
    
    print(f"Shape: {gdf.shape}")
    print(f"Columns: {list(gdf.columns)}")
    print(f"Geometry type: {gdf.geometry.geom_type.unique()}")
    print(f"CRS: {gdf.crs}")
    print(f"First few rows:")
    print(gdf.head())
    
    # Check for road class columns
    road_class_cols = [col for col in gdf.columns if any(road in col.upper() for road in ['CLASS', 'TYPE', 'CATEGORY'])]
    print(f"\nPotential road class columns: {road_class_cols}")
    
    # Check for name columns
    name_cols = [col for col in gdf.columns if any(name in col.upper() for name in ['NAME', 'STREET', 'ROAD'])]
    print(f"Potential name columns: {name_cols}")
    
    return gdf

def main():
    """Explore all datasets"""
    print("🔍 EXPLORING TORONTO ROAD SEGMENT CRASH DATA")
    print("=" * 80)
    
    try:
        # Explore collision data
        collision_df = explore_collision_data()
        
        # Explore KSI data
        ksi_df = explore_ksi_data()
        
        # Explore road network
        road_gdf = explore_road_network()
        
        print("\n" + "=" * 80)
        print("📋 SUMMARY")
        print("=" * 80)
        print(f"Collision data: {collision_df.shape[0]} records, {collision_df.shape[1]} columns")
        print(f"KSI data: {ksi_df.shape[0]} records, {ksi_df.shape[1]} columns")
        print(f"Road network: {road_gdf.shape[0]} segments, {road_gdf.shape[1]} columns")
        
        print("\nNext steps:")
        print("1. Update column names in config.py based on actual data")
        print("2. Update data_loader.py to use correct column names")
        print("3. Re-run the pipeline test")
        
    except Exception as e:
        print(f"❌ Error exploring data: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 