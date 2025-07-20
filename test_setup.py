"""
Test script to verify project setup and structure
"""

import os
from pathlib import Path

def test_project_structure():
    """Test that all required directories and files exist"""
    
    # Required directories
    required_dirs = [
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
    
    # Required files
    required_files = [
        "requirements.txt",
        "README.md",
        "config.py",
        "src/main.py",
        ".gitignore"
    ]
    
    print("Testing project structure...")
    print("=" * 50)
    
    # Check directories
    print("\nChecking directories:")
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"✅ {dir_path}")
        else:
            print(f"❌ {dir_path} - MISSING")
    
    # Check files
    print("\nChecking files:")
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - MISSING")
    
    # Check data directory is empty (ready for datasets)
    data_files = list(Path("data").glob("*"))
    if not data_files or all(f.name == ".gitkeep" for f in data_files):
        print("\n✅ Data directory is ready for datasets")
    else:
        print(f"\n⚠️  Data directory contains {len(data_files)} files")
    
    print("\n" + "=" * 50)
    print("Project structure test completed!")

if __name__ == "__main__":
    test_project_structure() 