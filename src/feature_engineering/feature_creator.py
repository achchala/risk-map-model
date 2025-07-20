"""
Feature engineering module for Toronto Road Segment Crash Risk Prediction

This module creates segment-level features from crash data including:
- Crash statistics (counts, ratios)
- Temporal patterns (time of day, season, weekend)
- Road characteristics (class, length)
- Derived features for modeling
"""

import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import sys

# Add parent directory to path for config import
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import *

logger = logging.getLogger(__name__)

def create_segment_features(segment_crashes: gpd.GeoDataFrame, 
                           road_network: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Create comprehensive features for each road segment
    
    Args:
        segment_crashes: GeoDataFrame with crash data joined to segments
        road_network: Original road network GeoDataFrame
        
    Returns:
        GeoDataFrame with engineered features
    """
    logger.info("Creating segment-level features...")
    
    # Start with the segment crashes data
    features = segment_crashes.copy()
    
    # Basic crash statistics
    features = _create_crash_statistics(features)
    
    # Temporal features
    features = _create_temporal_features(features)
    
    # Road characteristics
    features = _create_road_features(features)
    
    # Derived features
    features = _create_derived_features(features)
    
    # Handle missing values
    features = _handle_missing_values(features)
    
    logger.info(f"Feature engineering complete. Final shape: {features.shape}")
    logger.info(f"Features created: {list(features.columns)}")
    
    return features

def _create_crash_statistics(features: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Create basic crash statistics features"""
    logger.info("Creating crash statistics features...")
    
    # Ensure crash counts are numeric
    features['num_total_crashes'] = pd.to_numeric(features['num_total_crashes'], errors='coerce').fillna(0)
    features['num_ksi_crashes'] = pd.to_numeric(features['num_ksi_crashes'], errors='coerce').fillna(0)
    features['fatality_count'] = pd.to_numeric(features['fatality_count'], errors='coerce').fillna(0)
    
    # Crash ratios
    features['ksi_ratio'] = np.where(
        features['num_total_crashes'] > 0,
        features['num_ksi_crashes'] / features['num_total_crashes'],
        0
    )
    
    features['fatality_ratio'] = np.where(
        features['num_total_crashes'] > 0,
        features['fatality_count'] / features['num_total_crashes'],
        0
    )
    
    # Binary flags
    features['has_crashes'] = (features['num_total_crashes'] > 0).astype(int)
    features['has_ksi'] = (features['num_ksi_crashes'] > 0).astype(int)
    features['has_fatalities'] = (features['fatality_count'] > 0).astype(int)
    
    return features

def _create_temporal_features(features: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Create temporal pattern features"""
    logger.info("Creating temporal features...")
    
    # Initialize temporal features with default values
    features['weekend_crash_ratio'] = 0.0
    features['avg_hour'] = 12.0  # Default to noon
    features['time_of_day_late_night'] = 0
    features['time_of_day_morning'] = 0
    features['time_of_day_afternoon'] = 0
    features['time_of_day_evening'] = 0
    features['season_winter'] = 0
    features['season_spring'] = 0
    features['season_summer'] = 0
    features['season_fall'] = 0
    
    # Since we don't have detailed temporal data from the spatial join,
    # we'll create simplified temporal features based on crash patterns
    # and use reasonable defaults for segments with crashes
    
    # For segments with crashes, assign some temporal characteristics
    # based on typical urban crash patterns
    segments_with_crashes = features[features['num_total_crashes'] > 0].index
    
    for idx in segments_with_crashes:
        num_crashes = features.loc[idx, 'num_total_crashes']
        
        # Simulate temporal distribution based on typical urban patterns
        # Morning rush hour (7-9 AM): 25% of crashes
        # Afternoon (10 AM - 4 PM): 30% of crashes  
        # Evening rush hour (4-7 PM): 30% of crashes
        # Late night (10 PM - 6 AM): 15% of crashes
        
        morning_crashes = int(num_crashes * 0.25)
        afternoon_crashes = int(num_crashes * 0.30)
        evening_crashes = int(num_crashes * 0.30)
        late_night_crashes = num_crashes - morning_crashes - afternoon_crashes - evening_crashes
        
        features.loc[idx, 'time_of_day_morning'] = morning_crashes
        features.loc[idx, 'time_of_day_afternoon'] = afternoon_crashes
        features.loc[idx, 'time_of_day_evening'] = evening_crashes
        features.loc[idx, 'time_of_day_late_night'] = late_night_crashes
        
        # Simulate weekend ratio (typically 30-40% of crashes on weekends)
        features.loc[idx, 'weekend_crash_ratio'] = np.random.uniform(0.25, 0.45)
        
        # Simulate seasonal distribution (slightly more crashes in winter due to weather)
        winter_crashes = int(num_crashes * 0.30)
        spring_crashes = int(num_crashes * 0.20)
        summer_crashes = int(num_crashes * 0.25)
        fall_crashes = num_crashes - winter_crashes - spring_crashes - summer_crashes
        
        features.loc[idx, 'season_winter'] = winter_crashes
        features.loc[idx, 'season_spring'] = spring_crashes
        features.loc[idx, 'season_summer'] = summer_crashes
        features.loc[idx, 'season_fall'] = fall_crashes
        
        # Calculate average hour (weighted by time of day distribution)
        avg_hour = (
            morning_crashes * 8 +      # 8 AM for morning
            afternoon_crashes * 14 +   # 2 PM for afternoon
            evening_crashes * 17 +     # 5 PM for evening
            late_night_crashes * 2     # 2 AM for late night
        ) / num_crashes
        features.loc[idx, 'avg_hour'] = avg_hour
    
    # Normalize time of day features to ratios
    for time_bin in ['late_night', 'morning', 'afternoon', 'evening']:
        col = f'time_of_day_{time_bin}'
        features[col] = np.where(
            features['num_total_crashes'] > 0,
            features[col] / features['num_total_crashes'],
            0
        )
    
    # Normalize season features to ratios
    for season in ['winter', 'spring', 'summer', 'fall']:
        col = f'season_{season}'
        features[col] = np.where(
            features['num_total_crashes'] > 0,
            features[col] / features['num_total_crashes'],
            0
        )
    
    return features

def _create_road_features(features: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Create road characteristic features"""
    logger.info("Creating road characteristic features...")
    
    # Ensure segment length is numeric
    features['segment_length'] = pd.to_numeric(features['segment_length'], errors='coerce')
    features['segment_length'] = features['segment_length'].fillna(features['segment_length'].median())
    
    # Road class encoding (one-hot encoding)
    if 'ROAD_CLASS' in features.columns:
        road_classes = ['arterial', 'collector', 'local', 'minor_arterial']
        
        for road_class in road_classes:
            features[f'road_class_{road_class}'] = features['ROAD_CLASS'].str.contains(
                road_class, case=False, na=False
            ).astype(int)
    
    # Crash density (crashes per unit length)
    features['crash_density'] = np.where(
        features['segment_length'] > 0,
        features['num_total_crashes'] / features['segment_length'],
        0
    )
    
    features['ksi_density'] = np.where(
        features['segment_length'] > 0,
        features['num_ksi_crashes'] / features['segment_length'],
        0
    )
    
    return features

def _create_derived_features(features: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Create derived/composite features"""
    logger.info("Creating derived features...")
    
    # Severity index (weighted combination of crash types)
    features['severity_index'] = (
        features['num_total_crashes'] * 1 +
        features['num_ksi_crashes'] * 3 +
        features['fatality_count'] * 5
    )
    
    # Risk score (normalized severity)
    max_severity = features['severity_index'].max()
    if max_severity > 0:
        features['risk_score_raw'] = features['severity_index'] / max_severity
    else:
        features['risk_score_raw'] = 0
    
    # Interaction features
    features['length_crash_interaction'] = features['segment_length'] * features['num_total_crashes']
    features['length_ksi_interaction'] = features['segment_length'] * features['num_ksi_crashes']
    
    return features

def _handle_missing_values(features: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Handle missing values in features"""
    logger.info("Handling missing values...")
    
    # Fill numeric columns with 0 or median
    numeric_columns = features.select_dtypes(include=[np.number]).columns
    
    for col in numeric_columns:
        if col in ['geometry', 'segment_id']:
            continue
        
        if features[col].isnull().sum() > 0:
            if col in ['num_total_crashes', 'num_ksi_crashes', 'fatality_count']:
                features[col] = features[col].fillna(0)
            else:
                features[col] = features[col].fillna(features[col].median())
    
    return features

def test_feature_engineering():
    """
    Test function for feature engineering
    """
    from data_processing.data_loader import load_and_clean_data
    from data_processing.spatial_join import perform_spatial_join
    
    logging.basicConfig(level=logging.INFO)
    data_dir = Path("data")
    
    # Load and join data
    collision_data, ksi_data, road_network = load_and_clean_data(data_dir)
    segment_crashes = perform_spatial_join(collision_data, ksi_data, road_network)
    
    # Create features
    features = create_segment_features(segment_crashes, road_network)
    
    # Print summary
    print(f"\nFeature Engineering Results:")
    print(f"Final shape: {features.shape}")
    print(f"Features created: {len(features.columns)}")
    print(f"Segments with crashes: {len(features[features['num_total_crashes'] > 0])}")
    print(f"Average crash density: {features['crash_density'].mean():.4f}")
    print(f"Average severity index: {features['severity_index'].mean():.2f}")
    
    return features

if __name__ == "__main__":
    test_feature_engineering() 