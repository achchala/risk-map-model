#!/usr/bin/env python3
"""
Diagnostic script to analyze confidence score distribution
and identify why many scores are at 100%
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from collections import Counter

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from src.models.model_trainer import ModelTrainer

def analyze_confidence_distribution():
    """Analyze confidence scores from the model"""
    
    print("=" * 70)
    print("CONFIDENCE SCORE ANALYSIS")
    print("=" * 70)
    
    # Check if we have exported data
    geojson_file = Path("outputs/maps/toronto_road_risk.geojson")
    csv_file = Path("outputs/reports/road_risk_list.csv")
    
    if geojson_file.exists():
        print(f"\n1. Loading data from: {geojson_file}")
        data = gpd.read_file(geojson_file)
        print(f"   Loaded {len(data):,} segments")
        
        if 'confidence' in data.columns:
            confidence_scores = data['confidence'].values
            
            print(f"\n2. CONFIDENCE SCORE STATISTICS:")
            print(f"   Total segments: {len(confidence_scores):,}")
            print(f"   Mean confidence: {np.mean(confidence_scores):.4f} ({np.mean(confidence_scores)*100:.2f}%)")
            print(f"   Median confidence: {np.median(confidence_scores):.4f} ({np.median(confidence_scores)*100:.2f}%)")
            print(f"   Std deviation: {np.std(confidence_scores):.4f}")
            print(f"   Min confidence: {np.min(confidence_scores):.4f} ({np.min(confidence_scores)*100:.2f}%)")
            print(f"   Max confidence: {np.max(confidence_scores):.4f} ({np.max(confidence_scores)*100:.2f}%)")
            
            # Count exact 1.0 values
            exactly_100 = np.sum(confidence_scores == 1.0)
            exactly_100_pct = (exactly_100 / len(confidence_scores)) * 100
            
            # Count very high confidence (>0.99)
            very_high = np.sum(confidence_scores > 0.99)
            very_high_pct = (very_high / len(confidence_scores)) * 100
            
            # Count high confidence (>0.95)
            high_conf = np.sum(confidence_scores > 0.95)
            high_conf_pct = (high_conf / len(confidence_scores)) * 100
            
            print(f"\n3. CONFIDENCE DISTRIBUTION:")
            print(f"   Exactly 1.0 (100%): {exactly_100:,} segments ({exactly_100_pct:.2f}%)")
            print(f"   > 0.99 (>99%): {very_high:,} segments ({very_high_pct:.2f}%)")
            print(f"   > 0.95 (>95%): {high_conf:,} segments ({high_conf_pct:.2f}%)")
            
            # Distribution by bins
            print(f"\n4. CONFIDENCE BINS:")
            bins = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]
            for i in range(len(bins)-1):
                count = np.sum((confidence_scores >= bins[i]) & (confidence_scores < bins[i+1]))
                pct = (count / len(confidence_scores)) * 100
                print(f"   {bins[i]:.2f} - {bins[i+1]:.2f}: {count:,} segments ({pct:.2f}%)")
            # Count exactly 1.0 separately
            count_1 = np.sum(confidence_scores == 1.0)
            pct_1 = (count_1 / len(confidence_scores)) * 100
            print(f"   Exactly 1.0: {count_1:,} segments ({pct_1:.2f}%)")
            
            # Check if there are any unique values
            unique_values = np.unique(confidence_scores)
            print(f"\n5. UNIQUE CONFIDENCE VALUES:")
            print(f"   Number of unique values: {len(unique_values)}")
            if len(unique_values) <= 20:
                print(f"   Unique values: {sorted(unique_values)}")
            else:
                print(f"   First 10: {sorted(unique_values)[:10]}")
                print(f"   Last 10: {sorted(unique_values)[-10:]}")
            
            # Analyze by risk label
            if 'risk_label' in data.columns:
                print(f"\n6. CONFIDENCE BY RISK LABEL:")
                for label in ['low', 'medium', 'high']:
                    subset = confidence_scores[data['risk_label'] == label]
                    if len(subset) > 0:
                        exactly_100_subset = np.sum(subset == 1.0)
                        print(f"   {label.upper()} Risk:")
                        print(f"     Count: {len(subset):,}")
                        print(f"     Mean: {np.mean(subset):.4f} ({np.mean(subset)*100:.2f}%)")
                        print(f"     Exactly 1.0: {exactly_100_subset:,} ({exactly_100_subset/len(subset)*100:.2f}%)")
            
            # Check for potential issues
            print(f"\n7. POTENTIAL ISSUES:")
            issues = []
            
            if exactly_100_pct > 10:
                issues.append(f"⚠️  HIGH: {exactly_100_pct:.1f}% of predictions have exactly 100% confidence - this is unusual")
            
            if very_high_pct > 50:
                issues.append(f"⚠️  MEDIUM: {very_high_pct:.1f}% of predictions have >99% confidence - model may be overconfident")
            
            if np.std(confidence_scores) < 0.1:
                issues.append(f"⚠️  LOW: Very low standard deviation ({np.std(confidence_scores):.4f}) - confidence scores lack diversity")
            
            if len(unique_values) < 10:
                issues.append(f"⚠️  LOW: Only {len(unique_values)} unique confidence values - possible quantization issue")
            
            if len(issues) == 0:
                print("   [OK] No obvious issues detected")
            else:
                for issue in issues:
                    print(f"   {issue}")
            
            return data, confidence_scores
        else:
            print("   ⚠️  No 'confidence' column found in data")
            return None, None
    else:
        print(f"\n⚠️  No exported data found at {geojson_file}")
        print("   Run the pipeline first to generate confidence scores")
        return None, None

def analyze_model_probabilities():
    """Analyze raw model probabilities to understand the issue"""
    
    print(f"\n" + "=" * 70)
    print("MODEL PROBABILITY ANALYSIS")
    print("=" * 70)
    
    # Try to load a trained model
    model_file = Path("outputs/models/toronto_risk_model.joblib")
    
    if not model_file.exists():
        print(f"\n⚠️  No trained model found at {model_file}")
        print("   Run the pipeline first to train a model")
        return
    
        print(f"\n1. Loading model from: {model_file}")
    trainer = ModelTrainer()
    try:
        trainer.load_model(str(model_file))
        print("   [OK] Model loaded successfully")
    except Exception as e:
        print(f"   ✗ Error loading model: {e}")
        return
    
    # Try to load data and get some predictions
    geojson_file = Path("outputs/maps/toronto_road_risk.geojson")
    if geojson_file.exists():
        print(f"\n2. Loading data for prediction analysis...")
        data = gpd.read_file(geojson_file)
        
        # Sample a subset for analysis
        sample_size = min(1000, len(data))
        sample_data = data.sample(n=sample_size, random_state=42)
        
        print(f"   Analyzing {sample_size} sample segments...")
        
        try:
            # Prepare features
            X, _ = trainer.prepare_features(sample_data)
            X_scaled = trainer.scaler.transform(X)
            
            # Get probabilities
            probabilities = trainer.model.predict_proba(X_scaled)
            
            print(f"\n3. PROBABILITY ANALYSIS:")
            print(f"   Shape: {probabilities.shape}")
            print(f"   Classes: {trainer.label_encoder.classes_}")
            
            # Analyze probability distributions
            for i, class_name in enumerate(trainer.label_encoder.classes_):
                class_probs = probabilities[:, i]
                print(f"\n   {class_name.upper()} class probabilities:")
                print(f"     Mean: {np.mean(class_probs):.4f}")
                print(f"     Min: {np.min(class_probs):.4f}")
                print(f"     Max: {np.max(class_probs):.4f}")
                print(f"     Exactly 1.0: {np.sum(class_probs == 1.0):,} ({np.sum(class_probs == 1.0)/len(class_probs)*100:.2f}%)")
                print(f"     Exactly 0.0: {np.sum(class_probs == 0.0):,} ({np.sum(class_probs == 0.0)/len(class_probs)*100:.2f}%)")
            
            # Analyze max probabilities (confidence)
            max_probs = np.max(probabilities, axis=1)
            print(f"\n   MAX PROBABILITY (Confidence):")
            print(f"     Mean: {np.mean(max_probs):.4f}")
            print(f"     Exactly 1.0: {np.sum(max_probs == 1.0):,} ({np.sum(max_probs == 1.0)/len(max_probs)*100:.2f}%)")
            
            # Check if probabilities sum to 1
            prob_sums = np.sum(probabilities, axis=1)
            print(f"\n   PROBABILITY SUM CHECK:")
            print(f"     Mean sum: {np.mean(prob_sums):.4f}")
            print(f"     All sum to 1.0: {np.allclose(prob_sums, 1.0)}")
            if not np.allclose(prob_sums, 1.0):
                print(f"     ⚠️  WARNING: Probabilities don't sum to 1.0!")
                print(f"     Min sum: {np.min(prob_sums):.4f}")
                print(f"     Max sum: {np.max(prob_sums):.4f}")
            
            # Check for extreme probabilities
            extreme_cases = np.where((probabilities == 1.0).any(axis=1))[0]
            if len(extreme_cases) > 0:
                print(f"\n   EXTREME CASES (probability = 1.0):")
                print(f"     Found {len(extreme_cases)} cases with exactly 1.0 probability")
                print(f"     Sample probabilities:")
                for idx in extreme_cases[:5]:
                    print(f"       Segment {idx}: {probabilities[idx]}")
            
        except Exception as e:
            print(f"   ✗ Error analyzing probabilities: {e}")
            import traceback
            traceback.print_exc()

def main():
    """Main analysis function"""
    data, confidence_scores = analyze_confidence_distribution()
    analyze_model_probabilities()
    
    print(f"\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print("\nKEY FINDINGS:")
    print("1. Confidence is calculated as: max(probabilities, axis=1)")
    print("2. This means confidence = the highest class probability")
    print("3. If many scores are 100%, the model is very certain about predictions")
    print("4. This could indicate:")
    print("   - Overfitting (model too certain)")
    print("   - Strong feature discrimination")
    print("   - Random Forest behavior (trees strongly agree)")
    print("\nRECOMMENDATIONS:")
    print("- Consider probability calibration if overconfidence is an issue")
    print("- Check for data leakage in features")
    print("- Review model hyperparameters (max_depth, min_samples_leaf)")
    print("- Consider using calibrated probabilities (CalibratedClassifierCV)")

if __name__ == "__main__":
    main()
