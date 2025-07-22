from spatial_join_crashes import spatial_join_crashes
from data_loading import load_data
from feature_engineering import extract_features
import joblib

# Load and process data
roads, speed, lanes, road_class, ksi = load_data()
roads = extract_features(roads, speed, lanes, road_class)
roads = spatial_join_crashes(roads, ksi)

# If collision_risk is not present, define thresholds and add it
if "collision_risk" not in roads.columns:
    high_thresh = 10
    med_thresh = 3

    def risk_category(count):
        if count >= high_thresh:
            return "high"
        elif count >= med_thresh:
            return "medium"
        else:
            return "low"

    roads["collision_risk"] = roads["crash_count"].apply(risk_category)

print("collision_risk value counts:")
print(roads["collision_risk"].value_counts())
print("\ncrash_count summary:")
print(roads["crash_count"].describe())
