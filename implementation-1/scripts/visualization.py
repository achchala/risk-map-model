import geopandas as gpd
import matplotlib.pyplot as plt

# Load the processed roads file (after spatial join and modeling)
from spatial_join_crashes import spatial_join_crashes
from data_loading import load_data
from feature_engineering import extract_features

roads, speed, lanes, road_class, ksi = load_data()
roads = extract_features(roads, speed, lanes, road_class)
roads = spatial_join_crashes(roads, ksi)

# Optionally, add model predictions if available
# from model_random_forest import model
# roads['predicted_risk'] = model.predict(roads[['SPEED_LIMIT', 'NUMBER_OF_LANES', 'high_speed']].fillna(0))

# Plot crash counts
roads.plot(column="crash_count", cmap="Reds", legend=True, linewidth=0.5)
plt.title("Crash Count per Road Segment")
plt.show()

# Export to GeoPackage for QGIS
roads.to_file(
    "implementation-1/data/roads_with_crash_counts.gpkg", layer="roads", driver="GPKG"
)
print("Exported to implementation-1/data/roads_with_crash_counts.gpkg")
