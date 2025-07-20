import geopandas as gpd
import pandas as pd


def load_data():
    # Load road network
    roads = gpd.read_file("../data/ORN_data/ORN_ROAD_NET_ELEMENT.shp")
    # Load road attributes
    speed = pd.read_csv("../data/ORN_data/ORN_SPEED_LIMIT.csv")
    lanes = pd.read_csv("../data/ORN_data/ORN_NUMBER_OF_LANES.csv")
    road_class = pd.read_csv("../data/ORN_data/ORN_ROAD_CLASS.csv")
    # Load crash data
    ksi = pd.read_csv("../data/KSI.csv")
    return roads, speed, lanes, road_class, ksi


if __name__ == "__main__":
    roads, speed, lanes, road_class, ksi = load_data()
    print("Road segments:", len(roads))
    print("Speed records:", len(speed))
    print("Lanes records:", len(lanes))
    print("Road class records:", len(road_class))
    print("Crashes:", len(ksi))
