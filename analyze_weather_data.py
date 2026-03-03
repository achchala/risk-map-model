#!/usr/bin/env python3
"""
Analyze Toronto crash data to calculate weather-based risk multipliers

This script analyzes the KSI dataset to determine actual crash risk
by road surface condition (RDSFCOND) and visibility (VISIBILITY).
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT))

from config import *
from src.data_processing.data_loader import load_ksi_data

def analyze_weather_risk_multipliers():
    """
    Calculate risk multipliers from Toronto KSI data based on:
    - Road surface condition (RDSFCOND)
    - Visibility (VISIBILITY)
    """
    print("Loading Toronto KSI data...")
    data_dir = PROJECT_ROOT / "data"
    ksi_data = load_ksi_data(data_dir)
    
    print(f"\nTotal KSI records: {len(ksi_data):,}")
    
    # Check available weather columns
    weather_cols = ['VISIBILITY', 'RDSFCOND', 'LIGHT']
    available_cols = [col for col in weather_cols if col in ksi_data.columns]
    print(f"\nAvailable weather columns: {available_cols}")
    
    # Analyze Road Surface Condition
    if 'RDSFCOND' in ksi_data.columns:
        print("\n" + "="*60)
        print("ROAD SURFACE CONDITION ANALYSIS")
        print("="*60)
        
        # Count crashes by road condition
        road_cond_counts = ksi_data['RDSFCOND'].value_counts()
        print("\nCrash counts by road surface condition:")
        print(road_cond_counts)
        
        # Calculate crash rates (assuming we can normalize by exposure)
        # For now, calculate relative risk vs most common condition
        total_crashes = len(ksi_data)
        road_cond_rates = (road_cond_counts / total_crashes) * 100
        
        print("\nCrash rates (% of total):")
        print(road_cond_rates)
        
        # Find baseline (most common condition - likely "Dry")
        baseline_condition = road_cond_counts.index[0]
        baseline_rate = road_cond_rates[baseline_condition]
        
        print(f"\nBaseline condition: {baseline_condition} ({baseline_rate:.2f}% of crashes)")
        
        # Calculate relative risk multipliers
        print("\nRelative Risk Multipliers (vs baseline):")
        multipliers = {}
        for condition in road_cond_counts.index:
            condition_rate = road_cond_rates[condition]
            if baseline_rate > 0:
                multiplier = condition_rate / baseline_rate
                multipliers[condition] = multiplier
                print(f"  {condition:20s}: {multiplier:.2f}x ({condition_rate:.2f}% vs {baseline_rate:.2f}%)")
        
        # Map to weather conditions
        print("\n" + "-"*60)
        print("MAPPING TO WEATHER CONDITIONS:")
        print("-"*60)
        
        # Common mappings (adjust based on actual values in data)
        condition_mapping = {
            'Dry': 'clear',
            'Wet': 'rain',
            'Snow': 'snow',
            'Ice': 'sleet',
            'Slush': 'snow',
            'Loose Snow': 'snow',
            'Packed Snow': 'snow',
            'Standing Water': 'heavy_rain',
        }
        
        weather_multipliers = {}
        for road_cond, multiplier in multipliers.items():
            weather_type = condition_mapping.get(road_cond, road_cond.lower())
            if weather_type not in weather_multipliers or multiplier > weather_multipliers[weather_type]:
                weather_multipliers[weather_type] = multiplier
        
        print("\nSuggested weather multipliers from Toronto data:")
        for weather, mult in sorted(weather_multipliers.items(), key=lambda x: x[1], reverse=True):
            print(f"  {weather:20s}: {mult:.2f}")
    
    # Analyze Visibility
    if 'VISIBILITY' in ksi_data.columns:
        print("\n" + "="*60)
        print("VISIBILITY ANALYSIS")
        print("="*60)
        
        # Check data type and values
        print(f"\nVisibility column type: {ksi_data['VISIBILITY'].dtype}")
        print(f"Unique values: {ksi_data['VISIBILITY'].unique()[:20]}")  # Show first 20
        
        # If numeric, analyze by visibility ranges
        if pd.api.types.is_numeric_dtype(ksi_data['VISIBILITY']):
            visibility_ranges = [
                (0, 1, "Very Poor (<1km)"),
                (1, 3, "Poor (1-3km)"),
                (3, 5, "Reduced (3-5km)"),
                (5, float('inf'), "Good (>5km)")
            ]
            
            print("\nCrash rates by visibility range:")
            baseline_vis = None
            for min_vis, max_vis, label in visibility_ranges:
                if max_vis == float('inf'):
                    mask = ksi_data['VISIBILITY'] >= min_vis
                else:
                    mask = (ksi_data['VISIBILITY'] >= min_vis) & (ksi_data['VISIBILITY'] < max_vis)
                
                count = mask.sum()
                rate = (count / total_crashes) * 100
                print(f"  {label:25s}: {count:6,} crashes ({rate:.2f}%)")
                
                if baseline_vis is None:
                    baseline_vis = (min_vis, max_vis, label, rate)
            
            # Calculate multipliers
            if baseline_vis:
                baseline_rate = baseline_vis[3]
                print(f"\nVisibility multipliers (vs {baseline_vis[2]}):")
                for min_vis, max_vis, label in visibility_ranges:
                    if max_vis == float('inf'):
                        mask = ksi_data['VISIBILITY'] >= min_vis
                    else:
                        mask = (ksi_data['VISIBILITY'] >= min_vis) & (ksi_data['VISIBILITY'] < max_vis)
                    
                    count = mask.sum()
                    rate = (count / total_crashes) * 100
                    if baseline_rate > 0:
                        multiplier = rate / baseline_rate
                        print(f"  {label:25s}: {multiplier:.2f}x")
    
    # Analyze Light conditions
    if 'LIGHT' in ksi_data.columns:
        print("\n" + "="*60)
        print("LIGHTING CONDITIONS ANALYSIS")
        print("="*60)
        
        light_counts = ksi_data['LIGHT'].value_counts()
        print("\nCrash counts by lighting condition:")
        print(light_counts)
        
        light_rates = (light_counts / total_crashes) * 100
        print("\nCrash rates (% of total):")
        print(light_rates)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print("\nNext steps:")
    print("1. Review the multipliers calculated above")
    print("2. Compare with research-backed values")
    print("3. Update backend-api/app.py with calibrated values")
    print("4. Document the source of multipliers")

if __name__ == "__main__":
    analyze_weather_risk_multipliers()
