#!/usr/bin/env python3
"""
Fast test script for the Toronto Road Segment Crash Risk Prediction Pipeline

This script tests each component using the fast spatial join approach.
"""

import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_data_loading():
    """Test data loading step"""
    print("=" * 60)
    print("STEP 1: Testing Data Loading")
    print("=" * 60)
    
    try:
        from src.data_processing.data_loader import load_and_clean_data
        
        data_dir = Path("data")
        collision_data, ksi_data, road_network = load_and_clean_data(data_dir)
        
        print(f"✅ Data loading successful!")
        print(f"   - Collision data: {len(collision_data)} records")
        print(f"   - KSI data: {len(ksi_data)} records")
        print(f"   - Road network: {len(road_network)} segments")
        
        return collision_data, ksi_data, road_network
        
    except Exception as e:
        print(f"❌ Data loading failed: {str(e)}")
        return None, None, None

def test_spatial_join_fast(collision_data, ksi_data, road_network):
    """Test fast spatial join step"""
    print("\n" + "=" * 60)
    print("STEP 2: Testing Fast Spatial Join")
    print("=" * 60)
    
    try:
        from src.data_processing.spatial_join_fast import perform_spatial_join_fast
        
        segment_crashes = perform_spatial_join_fast(collision_data, ksi_data, road_network)
        
        print(f"✅ Fast spatial join successful!")
        print(f"   - Total segments: {len(segment_crashes)}")
        print(f"   - Segments with crashes: {len(segment_crashes[segment_crashes['num_total_crashes'] > 0])}")
        print(f"   - Segments with KSI: {len(segment_crashes[segment_crashes['num_ksi_crashes'] > 0])}")
        print(f"   - Total crashes: {segment_crashes['num_total_crashes'].sum()}")
        print(f"   - Total KSI: {segment_crashes['num_ksi_crashes'].sum()}")
        
        return segment_crashes
        
    except Exception as e:
        print(f"❌ Fast spatial join failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def test_feature_engineering(segment_crashes, road_network):
    """Test feature engineering step"""
    print("\n" + "=" * 60)
    print("STEP 3: Testing Feature Engineering")
    print("=" * 60)
    
    try:
        from src.feature_engineering.feature_creator import create_segment_features
        
        segment_features = create_segment_features(segment_crashes, road_network)
        
        print(f"✅ Feature engineering successful!")
        print(f"   - Final shape: {segment_features.shape}")
        print(f"   - Features created: {len(segment_features.columns)}")
        print(f"   - Segments with crashes: {len(segment_features[segment_features['num_total_crashes'] > 0])}")
        
        # Show some key features
        import pandas as pd
        key_features = ['num_total_crashes', 'num_ksi_crashes', 'crash_density', 'severity_index']
        for feature in key_features:
            if feature in segment_features.columns:
                mean_val = segment_features[feature].mean()
                if not pd.isna(mean_val):
                    print(f"   - {feature} (mean): {mean_val:.4f}")
        
        return segment_features
        
    except Exception as e:
        print(f"❌ Feature engineering failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def test_label_generation(segment_features):
    """Test label generation step"""
    print("\n" + "=" * 60)
    print("STEP 4: Testing Label Generation")
    print("=" * 60)
    
    try:
        from src.feature_engineering.label_generator import generate_risk_labels, analyze_label_distribution
        
        labeled_segments = generate_risk_labels(segment_features)
        analysis = analyze_label_distribution(labeled_segments)
        
        print(f"✅ Label generation successful!")
        print(f"   - Total segments: {len(labeled_segments)}")
        print(f"   - Label distribution: {analysis['label_counts']}")
        
        # Show characteristics by risk level
        for risk_level in ['low', 'medium', 'high']:
            if risk_level in analysis['label_counts']:
                count = analysis['label_counts'][risk_level]
                pct = analysis['label_percentages'][risk_level]
                print(f"   - {risk_level.capitalize()}: {count} segments ({pct:.1f}%)")
        
        return labeled_segments
        
    except Exception as e:
        print(f"❌ Label generation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Run all pipeline tests with fast approach"""
    print("🚀 Testing Toronto Road Segment Crash Risk Prediction Pipeline (Fast)")
    print("=" * 80)
    
    # Step 1: Data Loading
    collision_data, ksi_data, road_network = test_data_loading()
    if collision_data is None:
        print("\n❌ Pipeline failed at data loading step")
        return
    
    # Step 2: Fast Spatial Join
    segment_crashes = test_spatial_join_fast(collision_data, ksi_data, road_network)
    if segment_crashes is None:
        print("\n❌ Pipeline failed at spatial join step")
        return
    
    # Step 3: Feature Engineering
    segment_features = test_feature_engineering(segment_crashes, road_network)
    if segment_features is None:
        print("\n❌ Pipeline failed at feature engineering step")
        return
    
    # Step 4: Label Generation
    labeled_segments = test_label_generation(segment_features)
    if labeled_segments is None:
        print("\n❌ Pipeline failed at label generation step")
        return
    
    # Summary
    print("\n" + "=" * 80)
    print("🎉 ALL PIPELINE TESTS PASSED (Fast Approach)!")
    print("=" * 80)
    print("The data processing pipeline is working correctly.")
    print("Ready to proceed with model training and evaluation.")
    print("\nNext steps:")
    print("1. Implement model training module")
    print("2. Implement model evaluation module")
    print("3. Implement visualization modules")
    print("4. Run the complete pipeline")

if __name__ == "__main__":
    main() 