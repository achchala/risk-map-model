import pandas as pd
import geopandas as gpd


def extract_features(roads, speed, lanes, road_class, ksi):
    # Join road attributes (replace 'ID' with actual key)
    roads = roads.merge(speed, on="ID", how="left")
    roads = roads.merge(lanes, on="ID", how="left")
    roads = roads.merge(road_class, on="ID", how="left")

    # Example feature: high speed
    roads["high_speed"] = roads["SPEED_LIMIT"] > 50

    # Spatial join: count crashes per road segment
    crashes_gdf = gpd.GeoDataFrame(
        ksi,
        geometry=gpd.points_from_xy(ksi["LONGITUDE"], ksi["LATITUDE"]),
        crs=roads.crs,
    )
    joined = gpd.sjoin(crashes_gdf, roads, how="left", predicate="intersects")
    crash_counts = joined.groupby("ID").size().rename("crash_count")
    roads = roads.join(crash_counts, on="ID").fillna({"crash_count": 0})

    return roads


# Example usage
if __name__ == "__main__":
    from data_loading import load_data

    roads, speed, lanes, road_class, ksi = load_data()
    features = extract_features(roads, speed, lanes, road_class, ksi)
    print(features.head())
