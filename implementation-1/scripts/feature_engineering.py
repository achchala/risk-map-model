import pandas as pd
import geopandas as gpd


def extract_features(roads, speed, lanes, road_class):
    # Join speed limit
    roads = roads.merge(
        speed[["ORN_ROAD_NET_ELEMENT_ID", "SPEED_LIMIT"]],
        left_on="OGF_ID",
        right_on="ORN_ROAD_NET_ELEMENT_ID",
        how="left",
    )
    # Join number of lanes
    roads = roads.merge(
        lanes[["ORN_ROAD_NET_ELEMENT_ID", "NUMBER_OF_LANES"]],
        left_on="OGF_ID",
        right_on="ORN_ROAD_NET_ELEMENT_ID",
        how="left",
        suffixes=("", "_lanes"),
    )
    # Join road class
    roads = roads.merge(
        road_class[["ORN_ROAD_NET_ELEMENT_ID", "ROAD_CLASS"]],
        left_on="OGF_ID",
        right_on="ORN_ROAD_NET_ELEMENT_ID",
        how="left",
        suffixes=("", "_class"),
    )
    # Example feature: high speed
    roads["high_speed"] = roads["SPEED_LIMIT"].astype(float) > 50
    # Example feature: fill missing number of lanes with 1
    roads["NUMBER_OF_LANES"] = roads["NUMBER_OF_LANES"].fillna(1).astype(int)
    # Drop duplicate join columns
    roads = roads.drop(
        columns=[
            col
            for col in roads.columns
            if col.startswith("ORN_ROAD_NET_ELEMENT_ID") and col != "OGF_ID"
        ]
    )
    return roads


# Example usage
if __name__ == "__main__":
    from data_loading import load_data

    roads, speed, lanes, road_class, ksi = load_data()
    features = extract_features(roads, speed, lanes, road_class)
    print(features.head())
