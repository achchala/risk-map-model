"""
Label generator module for Toronto Road Segment Crash Risk Prediction

This module creates risk labels (Low/Medium/High) based on crash statistics
and KSI thresholds as defined in the project requirements.
"""

import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path
import logging
import sys

# Add parent directory to path for config import
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import *

logger = logging.getLogger(__name__)

def generate_risk_labels(segment_features: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Generate risk labels for road segments based on crash statistics
    
    Args:
        segment_features: GeoDataFrame with engineered features
        
    Returns:
        GeoDataFrame with risk labels added
    """
    logger.info("Generating risk labels for road segments...")
    
    # Create a copy to avoid modifying original
    labeled_segments = segment_features.copy()
    
    # Ensure crash counts are numeric
    labeled_segments['num_total_crashes'] = pd.to_numeric(
        labeled_segments['num_total_crashes'], errors='coerce'
    ).fillna(0)
    
    labeled_segments['num_ksi_crashes'] = pd.to_numeric(
        labeled_segments['num_ksi_crashes'], errors='coerce'
    ).fillna(0)
    
    # Generate risk labels based on rules
    labeled_segments['risk_label'] = labeled_segments.apply(
        _apply_risk_labeling_rules, axis=1
    )
    
    # Create numeric risk level for modeling
    risk_level_mapping = {'low': 0, 'medium': 1, 'high': 2}
    labeled_segments['risk_level'] = labeled_segments['risk_label'].map(risk_level_mapping)
    
    # Log label distribution
    label_counts = labeled_segments['risk_label'].value_counts()
    logger.info("Risk label distribution:")
    for label, count in label_counts.items():
        logger.info(f"  {label.capitalize()}: {count} segments ({count/len(labeled_segments)*100:.1f}%)")
    
    # Add fatality flag for modeling
    labeled_segments['fatality_flag'] = (labeled_segments['fatality_count'] > 0).astype(int)
    
    logger.info(f"Risk labeling complete. {len(labeled_segments)} segments labeled.")
    
    return labeled_segments

def _apply_risk_labeling_rules(row: pd.Series) -> str:
    """
    Apply risk labeling rules to a single road segment
    
    Args:
        row: Series containing segment data
        
    Returns:
        Risk label ('low', 'medium', 'high')
    """
    num_ksi = row['num_ksi_crashes']
    num_total = row['num_total_crashes']
    
    # High risk: KSI > 2 OR total crashes > 10
    if (num_ksi > RISK_LABELING_RULES['high']['ksi_threshold'] or 
        num_total > RISK_LABELING_RULES['high']['total_crashes_threshold']):
        return 'high'
    
    # Medium risk: KSI = 1 OR total crashes > 5
    elif (num_ksi >= RISK_LABELING_RULES['medium']['ksi_threshold'] or 
          num_total > RISK_LABELING_RULES['medium']['total_crashes_threshold']):
        return 'medium'
    
    # Low risk: everything else
    else:
        return 'low'

def analyze_label_distribution(labeled_segments: gpd.GeoDataFrame) -> dict:
    """
    Analyze the distribution of risk labels and their characteristics
    
    Args:
        labeled_segments: GeoDataFrame with risk labels
        
    Returns:
        Dictionary with analysis results
    """
    logger.info("Analyzing risk label distribution...")
    
    analysis = {}
    
    # Basic distribution
    analysis['label_counts'] = labeled_segments['risk_label'].value_counts().to_dict()
    analysis['label_percentages'] = (
        labeled_segments['risk_label'].value_counts(normalize=True) * 100
    ).to_dict()
    
    # Characteristics by risk level
    for risk_level in ['low', 'medium', 'high']:
        subset = labeled_segments[labeled_segments['risk_label'] == risk_level]
        
        analysis[f'{risk_level}_characteristics'] = {
            'count': len(subset),
            'avg_total_crashes': subset['num_total_crashes'].mean(),
            'avg_ksi_crashes': subset['num_ksi_crashes'].mean(),
            'avg_fatality_count': subset['fatality_count'].mean(),
            'avg_segment_length': subset['segment_length'].mean(),
            'avg_crash_density': subset['crash_density'].mean(),
            'avg_severity_index': subset['severity_index'].mean(),
            'has_crashes_pct': subset['has_crashes'].mean() * 100,
            'has_ksi_pct': subset['has_ksi'].mean() * 100,
            'has_fatalities_pct': subset['has_fatalities'].mean() * 100
        }
    
    # Road class distribution by risk level
    road_class_cols = [col for col in labeled_segments.columns if col.startswith('road_class_')]
    for risk_level in ['low', 'medium', 'high']:
        subset = labeled_segments[labeled_segments['risk_label'] == risk_level]
        analysis[f'{risk_level}_road_classes'] = {}
        
        for col in road_class_cols:
            road_class = col.replace('road_class_', '')
            analysis[f'{risk_level}_road_classes'][road_class] = subset[col].mean() * 100
    
    logger.info("Risk label analysis complete.")
    return analysis

def validate_labels(labeled_segments: gpd.GeoDataFrame) -> bool:
    """
    Validate that risk labels are correctly assigned
    
    Args:
        labeled_segments: GeoDataFrame with risk labels
        
    Returns:
        True if validation passes, False otherwise
    """
    logger.info("Validating risk labels...")
    
    validation_passed = True
    
    # Check that all segments have labels
    if labeled_segments['risk_label'].isnull().any():
        logger.error("Some segments are missing risk labels")
        validation_passed = False
    
    # Check that labels are valid
    valid_labels = {'low', 'medium', 'high'}
    invalid_labels = set(labeled_segments['risk_label'].unique()) - valid_labels
    if invalid_labels:
        logger.error(f"Invalid risk labels found: {invalid_labels}")
        validation_passed = False
    
    # Check that high-risk segments meet criteria
    high_risk = labeled_segments[labeled_segments['risk_label'] == 'high']
    for idx, row in high_risk.iterrows():
        num_ksi = row['num_ksi_crashes']
        num_total = row['num_total_crashes']
        
        if not (num_ksi > RISK_LABELING_RULES['high']['ksi_threshold'] or 
                num_total > RISK_LABELING_RULES['high']['total_crashes_threshold']):
            logger.error(f"High-risk segment {idx} doesn't meet criteria: KSI={num_ksi}, Total={num_total}")
            validation_passed = False
    
    # Check that low-risk segments don't meet higher criteria
    low_risk = labeled_segments[labeled_segments['risk_label'] == 'low']
    for idx, row in low_risk.iterrows():
        num_ksi = row['num_ksi_crashes']
        num_total = row['num_total_crashes']
        
        if (num_ksi > RISK_LABELING_RULES['high']['ksi_threshold'] or 
            num_total > RISK_LABELING_RULES['high']['total_crashes_threshold']):
            logger.error(f"Low-risk segment {idx} meets high-risk criteria: KSI={num_ksi}, Total={num_total}")
            validation_passed = False
    
    if validation_passed:
        logger.info("Risk label validation passed!")
    else:
        logger.error("Risk label validation failed!")
    
    return validation_passed

def test_label_generation():
    """
    Test function for label generation
    """
    from data_processing.data_loader import load_and_clean_data
    from data_processing.spatial_join_fast import perform_spatial_join_fast
    from feature_creator import create_segment_features
    
    logging.basicConfig(level=logging.INFO)
    data_dir = Path("data")
    
    # Load and process data
    collision_data, ksi_data, road_network = load_and_clean_data(data_dir)
    segment_crashes = perform_spatial_join_fast(collision_data, ksi_data, road_network)
    segment_features = create_segment_features(segment_crashes, road_network)
    
    # Generate labels
    labeled_segments = generate_risk_labels(segment_features)
    
    # Analyze distribution
    analysis = analyze_label_distribution(labeled_segments)
    
    # Validate labels
    validation_passed = validate_labels(labeled_segments)
    
    # Print summary
    print(f"\nLabel Generation Results:")
    print(f"Total segments: {len(labeled_segments)}")
    print(f"Label distribution: {analysis['label_counts']}")
    print(f"Validation passed: {validation_passed}")
    
    return labeled_segments, analysis

if __name__ == "__main__":
    test_label_generation() 