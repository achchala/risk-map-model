"""
Configuration file for Toronto Road Segment Crash Risk Prediction MVP

This file contains all the parameters, file paths, and settings used throughout the pipeline.
"""

from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = PROJECT_ROOT / "models"

# Data file names
COLLISION_DATA_FILE = "Traffic_Collisions_Open_Data_2437597425626428496.xlsx"
KSI_DATA_FILE = "TOTAL_KSI_6386614326836635957.csv"
ROAD_NETWORK_FILE = "Centreline - Version 2 - 4326.geojson"

# Column mappings for different datasets
COLLISION_COLUMNS = {
    'latitude': 'LAT_WGS84',
    'longitude': 'LONG_WGS84', 
    'date': 'OCC_DATE',
    'time': 'OCC_HOUR',
    'injury': 'INJURY_COLLISIONS',
    'fatalities': 'FATALITIES'
}

KSI_COLUMNS = {
    'latitude': 'LATITUDE',
    'longitude': 'LONGITUDE',
    'date': 'DATE', 
    'time': 'TIME',
    'injury': 'INJURY',
    'fatalities': 'FATAL_NO'
}

ROAD_COLUMNS = {
    'road_class': 'LINEAR_NAME_TYPE',
    'name': 'LINEAR_NAME'
}

# Spatial processing parameters
SPATIAL_BUFFER_DISTANCE = 20  # meters - for joining crashes to road segments
CRS_EPSG = 4326  # WGS84 coordinate system

# Feature engineering parameters
TIME_BINS = {
    'late_night': (0, 5),
    'morning': (6, 11),
    'afternoon': (12, 17),
    'evening': (18, 23)
}

SEASON_MAPPING = {
    12: 'winter', 1: 'winter', 2: 'winter',
    3: 'spring', 4: 'spring', 5: 'spring',
    6: 'summer', 7: 'summer', 8: 'summer',
    9: 'fall', 10: 'fall', 11: 'fall'
}

# Risk labeling rules
RISK_LABELING_RULES = {
    'high': {
        'ksi_threshold': 2,
        'total_crashes_threshold': 10
    },
    'medium': {
        'ksi_threshold': 1,
        'total_crashes_threshold': 5
    }
    # 'low' is default (everything else)
}

# Model parameters
MODEL_CONFIG = {
    'algorithm': 'random_forest',
    'n_estimators': 100,
    'max_depth': 10,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'random_state': 42,
    'class_weight': 'balanced'
}

# Cross-validation parameters
CV_CONFIG = {
    'n_splits': 5,
    'random_state': 42,
    'shuffle': True
}

# Feature columns to use for modeling
FEATURE_COLUMNS = [
    'num_total_crashes',
    'num_ksi_crashes',
    'fatality_flag',
    'weekend_crash_ratio',
    'avg_hour',
    'segment_length',
    'road_class_arterial',
    'road_class_collector',
    'road_class_local',
    'road_class_minor_arterial',
    'time_of_day_morning',
    'time_of_day_afternoon',
    'time_of_day_evening',
    'time_of_day_late_night',
    'season_winter',
    'season_spring',
    'season_summer',
    'season_fall'
]

# Output file names
OUTPUT_FILES = {
    'risk_segments': 'risk_segments.geojson',
    'model': 'risk_model.joblib',
    'feature_importance': 'feature_importance.png',
    'confusion_matrix': 'confusion_matrix.png',
    'risk_map': 'risk_map.html',
    'analysis_report': 'analysis_report.html'
}

# Visualization parameters
MAP_CONFIG = {
    'center_lat': 43.6532,
    'center_lon': -79.3832,
    'zoom': 10,
    'tiles': 'OpenStreetMap'
}

# Logging configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': 'pipeline.log'
} 