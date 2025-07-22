import geopandas as gpd
from spatial_join_crashes import spatial_join_crashes
from data_loading import load_data
from feature_engineering import extract_features
import numpy as np
import joblib

# Load and process data
roads, speed, lanes, road_class, ksi = load_data()
roads = extract_features(roads, speed, lanes, road_class)
roads = spatial_join_crashes(roads, ksi)

# Load models
rf_model = joblib.load("implementation-1/data/model_random_forest.joblib")
poisson_model = joblib.load("implementation-1/data/model_poisson.joblib")

feature_cols = ["SPEED_LIMIT", "NUMBER_OF_LANES", "high_speed"]
roads["rf_pred"] = rf_model.predict(roads[feature_cols].fillna(0))
roads["poisson_pred"] = poisson_model.predict(roads[feature_cols].fillna(0))

# Define thresholds (customize as needed)
high_thresh = 5
med_thresh = 1


def risk_category(count):
    if count >= high_thresh:
        return "high"
    elif count >= med_thresh:
        return "medium"
    else:
        return "low"


roads["collision_risk"] = roads["crash_count"].apply(risk_category)

# Export to GeoJSON
roads.to_file("implementation-1/data/roads_with_crash_counts.geojson", driver="GeoJSON")
print("Exported to implementation-1/data/roads_with_crash_counts.geojson")
