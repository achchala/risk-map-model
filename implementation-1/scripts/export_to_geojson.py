import geopandas as gpd
from spatial_join_crashes import spatial_join_crashes
from data_loading import load_data
from feature_engineering import extract_features

# Load and process data
roads, speed, lanes, road_class, ksi = load_data()
roads = extract_features(roads, speed, lanes, road_class)
roads = spatial_join_crashes(roads, ksi)

# Optionally, add model predictions if available
# from model_random_forest import model
# roads['predicted_risk'] = model.predict(roads[['SPEED_LIMIT', 'NUMBER_OF_LANES', 'high_speed']].fillna(0))

# Export to GeoJSON
roads.to_file("implementation-1/data/roads_with_crash_counts.geojson", driver="GeoJSON")
print("Exported to implementation-1/data/roads_with_crash_counts.geojson")
