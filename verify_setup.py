#!/usr/bin/env python3
"""
Simple verification script for project setup
"""

import os
from pathlib import Path

def main():
    print("🔍 Verifying Toronto Road Segment Crash Risk Prediction Project Setup")
    print("=" * 70)
    
    # Check main directories
    dirs_to_check = [
        "data",
        "src", 
        "src/data_processing",
        "src/feature_engineering", 
        "src/modeling",
        "src/visualization",
        "models",
        "outputs",
        "outputs/maps",
        "outputs/models",
        "outputs/reports",
        "notebooks",
        "docs"
    ]
    
    print("\n📁 Checking directories:")
    all_dirs_ok = True
    for dir_path in dirs_to_check:
        if Path(dir_path).exists():
            print(f"  ✅ {dir_path}")
        else:
            print(f"  ❌ {dir_path} - MISSING")
            all_dirs_ok = False
    
    # Check main files
    files_to_check = [
        "requirements.txt",
        "README.md", 
        "config.py",
        "src/main.py",
        ".gitignore"
    ]
    
    print("\n📄 Checking files:")
    all_files_ok = True
    for file_path in files_to_check:
        if Path(file_path).exists():
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path} - MISSING")
            all_files_ok = False
    
    # Check data directory is ready
    data_files = list(Path("data").glob("*"))
    if not data_files or all(f.name == ".gitkeep" for f in data_files):
        print("\n📂 Data directory is ready for your datasets!")
        print("   Expected files:")
        print("   - Traffic_Collisions_Open_Data_2437597425626428496.xlsx")
        print("   - TOTAL_KSI_6386614326836635957.csv") 
        print("   - Centreline - Version 2 - 4326.geojson")
    else:
        print(f"\n⚠️  Data directory contains {len(data_files)} files")
    
    print("\n" + "=" * 70)
    
    if all_dirs_ok and all_files_ok:
        print("🎉 Project structure is complete and ready!")
        print("\nNext steps:")
        print("1. Add your 3 datasets to the 'data/' folder")
        print("2. Install dependencies: py -m pip install -r requirements.txt")
        print("3. Run the pipeline: py src/main.py")
    else:
        print("⚠️  Some components are missing. Please check the structure.")
    
    print("=" * 70)

if __name__ == "__main__":
    main() 