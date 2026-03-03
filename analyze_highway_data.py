#!/usr/bin/env python3
"""
Analysis script to identify highway information in the road network dataset

This script analyzes the road network GeoJSON to determine:
1. What values exist in LINEAR_NAME_TYPE (road classification)
2. Whether FEATURE_CODE or FEATURE_CODE_DESC contain highway information
3. Whether road names (LINEAR_NAME) contain highway indicators
4. Summary of highway identification capabilities
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
from collections import Counter
import re

# Add project root to path
sys.path.append(str(Path(__file__).parent))
from config import DATA_DIR, ROAD_NETWORK_FILE, ROAD_COLUMNS

def analyze_road_network_structure():
    """Analyze the structure and available fields in the road network dataset"""
    
    print("=" * 70)
    print("HIGHWAY IDENTIFICATION ANALYSIS")
    print("=" * 70)
    
    # Load road network
    file_path = DATA_DIR / ROAD_NETWORK_FILE
    if not file_path.exists():
        print(f"\n⚠️  Road network file not found: {file_path}")
        print("   Please ensure the data file is in the data/ directory")
        return None
    
    print(f"\n1. Loading road network from: {file_path}")
    try:
        gdf = gpd.read_file(file_path)
        print(f"   [OK] Loaded {len(gdf):,} road segments")
    except Exception as e:
        print(f"   ✗ Error loading file: {e}")
        return None
    
    # Display dataset structure
    print(f"\n2. DATASET STRUCTURE:")
    print(f"   Total columns: {len(gdf.columns)}")
    print(f"   Total segments: {len(gdf):,}")
    print(f"   Geometry type: {gdf.geometry.type.iloc[0] if len(gdf) > 0 else 'N/A'}")
    
    # List all available columns
    print(f"\n3. AVAILABLE COLUMNS:")
    relevant_columns = [
        'LINEAR_NAME_TYPE', 'LINEAR_NAME', 'FEATURE_CODE', 'FEATURE_CODE_DESC',
        'ROAD_CLASS', 'LINEAR_NAME_FULL', 'LINEAR_NAME_DIR', 'LINEAR_NAME_DESC'
    ]
    for col in gdf.columns:
        if any(relevant in col.upper() for relevant in ['LINEAR', 'FEATURE', 'ROAD', 'NAME', 'TYPE', 'CLASS']):
            print(f"   ✓ {col}")
        elif col == 'geometry':
            print(f"   - {col} (geometry)")
        elif len(gdf.columns) <= 20:
            print(f"   - {col}")
    
    return gdf

def analyze_linear_name_type(gdf):
    """Analyze LINEAR_NAME_TYPE field for road classification"""
    
    print(f"\n" + "=" * 70)
    print("LINEAR_NAME_TYPE ANALYSIS (Road Classification)")
    print("=" * 70)
    
    road_class_col = ROAD_COLUMNS['road_class']
    
    if road_class_col not in gdf.columns:
        print(f"\n⚠️  Column '{road_class_col}' not found in dataset")
        print(f"   Available columns: {list(gdf.columns)[:10]}")
        return None
    
    print(f"\n1. Analyzing '{road_class_col}' field...")
    
    # Get unique values and counts
    value_counts = gdf[road_class_col].value_counts()
    total_segments = len(gdf)
    non_null_count = gdf[road_class_col].notna().sum()
    null_count = gdf[road_class_col].isna().sum()
    
    print(f"\n2. VALUE DISTRIBUTION:")
    print(f"   Total segments: {total_segments:,}")
    print(f"   Non-null values: {non_null_count:,} ({non_null_count/total_segments*100:.1f}%)")
    print(f"   Null values: {null_count:,} ({null_count/total_segments*100:.1f}%)")
    
    print(f"\n3. UNIQUE VALUES (Top 20):")
    for value, count in value_counts.head(20).items():
        pct = (count / total_segments) * 100
        print(f"   {str(value):<40} {count:>8,} ({pct:>5.2f}%)")
    
    if len(value_counts) > 20:
        print(f"   ... and {len(value_counts) - 20} more unique values")
    
    # Check for highway-related keywords
    print(f"\n4. HIGHWAY KEYWORD SEARCH:")
    highway_keywords = ['highway', 'expressway', 'freeway', 'parkway', 'turnpike', 'autoroute']
    highway_matches = {}
    
    for keyword in highway_keywords:
        # Case-insensitive search
        matches = gdf[road_class_col].astype(str).str.contains(
            keyword, case=False, na=False, regex=False
        )
        count = matches.sum()
        if count > 0:
            highway_matches[keyword] = count
            pct = (count / total_segments) * 100
            print(f"   '{keyword}': {count:,} segments ({pct:.2f}%)")
    
    if len(highway_matches) == 0:
        print("   [No highway keywords found in LINEAR_NAME_TYPE]")
    
    # Show sample values containing highway keywords
    if len(highway_matches) > 0:
        print(f"\n5. SAMPLE VALUES WITH HIGHWAY KEYWORDS:")
        for keyword in list(highway_matches.keys())[:3]:
            matches = gdf[gdf[road_class_col].astype(str).str.contains(
                keyword, case=False, na=False, regex=False
            )]
            if len(matches) > 0:
                sample_values = matches[road_class_col].unique()[:5]
                for val in sample_values:
                    print(f"   - {val}")
    
    return value_counts, highway_matches

def analyze_feature_code(gdf):
    """Analyze FEATURE_CODE and FEATURE_CODE_DESC fields"""
    
    print(f"\n" + "=" * 70)
    print("FEATURE_CODE ANALYSIS")
    print("=" * 70)
    
    feature_code_col = 'FEATURE_CODE'
    feature_desc_col = 'FEATURE_CODE_DESC'
    
    has_feature_code = feature_code_col in gdf.columns
    has_feature_desc = feature_desc_col in gdf.columns
    
    if not has_feature_code and not has_feature_desc:
        print(f"\n⚠️  Neither '{feature_code_col}' nor '{feature_desc_col}' found in dataset")
        return None, None
    
    results = {}
    
    if has_feature_code:
        print(f"\n1. Analyzing '{feature_code_col}' field...")
        value_counts = gdf[feature_code_col].value_counts()
        total = len(gdf)
        non_null = gdf[feature_code_col].notna().sum()
        
        print(f"   Total segments: {total:,}")
        print(f"   Non-null values: {non_null:,} ({non_null/total*100:.1f}%)")
        print(f"\n   UNIQUE VALUES (Top 15):")
        for value, count in value_counts.head(15).items():
            pct = (count / total) * 100
            print(f"   {str(value):<20} {count:>8,} ({pct:>5.2f}%)")
        
        results['feature_code'] = value_counts
    
    if has_feature_desc:
        print(f"\n2. Analyzing '{feature_desc_col}' field...")
        value_counts = gdf[feature_desc_col].value_counts()
        total = len(gdf)
        non_null = gdf[feature_desc_col].notna().sum()
        
        print(f"   Total segments: {total:,}")
        print(f"   Non-null values: {non_null:,} ({non_null/total*100:.1f}%)")
        print(f"\n   UNIQUE VALUES (Top 15):")
        for value, count in value_counts.head(15).items():
            pct = (count / total) * 100
            print(f"   {str(value):<50} {count:>8,} ({pct:>5.2f}%)")
        
        # Check for highway keywords in descriptions
        print(f"\n3. HIGHWAY KEYWORD SEARCH IN DESCRIPTIONS:")
        highway_keywords = ['highway', 'expressway', 'freeway', 'parkway', 'turnpike']
        for keyword in highway_keywords:
            matches = gdf[feature_desc_col].astype(str).str.contains(
                keyword, case=False, na=False, regex=False
            )
            count = matches.sum()
            if count > 0:
                pct = (count / total) * 100
                print(f"   '{keyword}': {count:,} segments ({pct:.2f}%)")
        
        results['feature_desc'] = value_counts
    
    return results.get('feature_code'), results.get('feature_desc')

def analyze_road_names(gdf):
    """Analyze LINEAR_NAME field for highway indicators"""
    
    print(f"\n" + "=" * 70)
    print("ROAD NAME ANALYSIS (LINEAR_NAME)")
    print("=" * 70)
    
    name_col = ROAD_COLUMNS['name']
    
    if name_col not in gdf.columns:
        print(f"\n⚠️  Column '{name_col}' not found in dataset")
        return None
    
    print(f"\n1. Analyzing '{name_col}' field...")
    
    total = len(gdf)
    non_null = gdf[name_col].notna().sum()
    print(f"   Total segments: {total:,}")
    print(f"   Non-null values: {non_null:,} ({non_null/total*100:.1f}%)")
    
    # Search for highway indicators in road names
    print(f"\n2. HIGHWAY INDICATORS IN ROAD NAMES:")
    
    # Common highway patterns
    highway_patterns = {
        'Highway Number': r'\b(Hwy|Highway|HWY)\s*(\d{1,4})\b',
        '400-series': r'\b(40[0-9]|4[1-9][0-9])\b',  # 400-499
        'QEW': r'\bQEW\b',
        'Expressway': r'\b(Expressway|EXPWY)\b',
        'Freeway': r'\bFreeway\b',
        'Parkway': r'\bParkway\b',
        'Gardiner': r'\bGardiner\b',  # Toronto-specific
        'Don Valley': r'\bDon Valley\b',  # Toronto-specific
    }
    
    matches_summary = {}
    
    for pattern_name, pattern in highway_patterns.items():
        matches = gdf[name_col].astype(str).str.contains(
            pattern, case=False, na=False, regex=True
        )
        count = matches.sum()
        if count > 0:
            matches_summary[pattern_name] = count
            pct = (count / total) * 100
            print(f"   {pattern_name:<20}: {count:>6,} segments ({pct:>5.2f}%)")
    
    if len(matches_summary) == 0:
        print("   [No highway patterns found in road names]")
    
    # Show sample road names
    if len(matches_summary) > 0:
        print(f"\n3. SAMPLE ROAD NAMES WITH HIGHWAY INDICATORS:")
        # Combine all patterns
        combined_pattern = '|'.join([p for p in highway_patterns.values()])
        matching_segments = gdf[gdf[name_col].astype(str).str.contains(
            combined_pattern, case=False, na=False, regex=True
        )]
        
        if len(matching_segments) > 0:
            sample_names = matching_segments[name_col].unique()[:10]
            for name in sample_names:
                print(f"   - {name}")
    
    return matches_summary

def generate_summary_report(gdf, linear_name_type_counts, highway_matches, 
                           feature_code_counts, feature_desc_counts, name_matches):
    """Generate a comprehensive summary report"""
    
    print(f"\n" + "=" * 70)
    print("SUMMARY REPORT")
    print("=" * 70)
    
    print(f"\n1. HIGHWAY IDENTIFICATION CAPABILITIES:")
    
    can_identify = False
    methods = []
    
    # Check LINEAR_NAME_TYPE
    if highway_matches and len(highway_matches) > 0:
        can_identify = True
        methods.append(f"LINEAR_NAME_TYPE: {sum(highway_matches.values()):,} segments match highway keywords")
    
    # Check FEATURE_CODE_DESC
    if feature_desc_counts is not None:
        highway_in_desc = feature_desc_counts[
            feature_desc_counts.index.astype(str).str.contains(
                'highway|expressway|freeway', case=False, na=False, regex=True
            )
        ]
        if len(highway_in_desc) > 0:
            can_identify = True
            methods.append(f"FEATURE_CODE_DESC: Contains highway classifications")
    
    # Check road names
    if name_matches and len(name_matches) > 0:
        can_identify = True
        methods.append(f"LINEAR_NAME: {sum(name_matches.values()):,} segments have highway indicators in names")
    
    if can_identify:
        print("   ✓ YES - Highways can be identified through:")
        for method in methods:
            print(f"     • {method}")
    else:
        print("   ✗ NO - No clear highway identification found in standard fields")
        print("     However, highways may still be identifiable through:")
        print("     • Road name patterns (e.g., 'Highway 401', 'QEW')")
        print("     • Road class values (may need manual inspection)")
        print("     • Feature codes (may need documentation)")
    
    print(f"\n2. RECOMMENDATIONS:")
    
    if highway_matches and len(highway_matches) > 0:
        print("   • Add 'highway' to road_class list in feature_creator.py")
        print("   • Create road_class_highway feature for model training")
    elif name_matches and len(name_matches) > 0:
        print("   • Consider creating a derived 'is_highway' feature based on road name patterns")
        print("   • Use regex patterns to identify highways from LINEAR_NAME field")
    else:
        print("   • Manual inspection of LINEAR_NAME_TYPE values may reveal highway classifications")
        print("   • Check Toronto Open Data documentation for FEATURE_CODE meanings")
        print("   • Consider using road name patterns as a fallback method")
    
    print(f"\n3. NEXT STEPS:")
    print("   • Review the detailed analysis above")
    print("   • If highways are found, update feature_creator.py to include highway classification")
    print("   • Add highway as a road class feature for the ML model")
    print("   • Consider creating a separate highway risk analysis")

def main():
    """Main analysis function"""
    
    # Analyze dataset structure
    gdf = analyze_road_network_structure()
    if gdf is None:
        return
    
    # Analyze LINEAR_NAME_TYPE
    linear_name_type_counts, highway_matches = analyze_linear_name_type(gdf)
    
    # Analyze FEATURE_CODE fields
    feature_code_counts, feature_desc_counts = analyze_feature_code(gdf)
    
    # Analyze road names
    name_matches = analyze_road_names(gdf)
    
    # Generate summary
    generate_summary_report(
        gdf, linear_name_type_counts, highway_matches,
        feature_code_counts, feature_desc_counts, name_matches
    )
    
    print(f"\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nOutput saved. Review the analysis above to determine highway identification strategy.")

if __name__ == "__main__":
    main()
