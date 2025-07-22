import geopandas as gpd
import pandas as pd
import os
from shapely.geometry import LineString


def drop_z(geom):
    if hasattr(geom, "has_z") and geom.has_z:
        return LineString([(x, y) for x, y, *_ in geom.coords])
    return geom


def spatial_join_crashes(roads, ksi):
    # Ensure all geometries are 2D and valid
    roads = roads[roads.is_valid & roads.geometry.notnull()].copy()
    roads["geometry"] = roads["geometry"].apply(drop_z)
    # Ensure CRS matches
    if roads.crs is None:
        roads = roads.set_crs("EPSG:4326")
    # Create GeoDataFrame for crashes
    crashes_gdf = gpd.GeoDataFrame(
        ksi,
        geometry=gpd.points_from_xy(ksi["LONGITUDE"], ksi["LATITUDE"]),
        crs=roads.crs,
    )
    # Buffer road segments by 10 meters in projected CRS
    roads_proj = roads.to_crs("EPSG:32617")
    roads_proj["geometry"] = roads_proj.geometry.buffer(10)
    roads_buffered = roads_proj.to_crs(roads.crs)
    # Spatial join: assign each crash to a road segment
    joined = gpd.sjoin(crashes_gdf, roads_buffered, how="left", predicate="within")
    # Count crashes per segment
    crash_counts = joined.groupby("OGF_ID").size().rename("crash_count")
    # Add crash_count to roads
    roads = roads.join(crash_counts, on="OGF_ID")
    roads["crash_count"] = roads["crash_count"].fillna(0).astype(int)
    return roads


if __name__ == "__main__":
    from data_loading import load_data
    from feature_engineering import extract_features

    roads, speed, lanes, road_class, ksi = load_data()
    roads = extract_features(roads, speed, lanes, road_class)
    roads_with_crashes = spatial_join_crashes(roads, ksi)
    print(roads_with_crashes[["OGF_ID", "crash_count"]].head())

    # Assuming roads_with_crashes is your GeoDataFrame
    has_crashes = roads_with_crashes[roads_with_crashes["crash_count"] > 0]
    print(
        has_crashes[["OGF_ID", "crash_count"]].head(10)
    )  # Show first 10 segments with crashes
    print(f"Total segments with at least one crash: {len(has_crashes)}")
