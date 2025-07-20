import geopandas as gpd
import pandas as pd
import os


def load_data():
    # Set base directory to implementation-1
    base = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(base, "data/ORN_data")
    # Load road network
    roads = gpd.read_file(os.path.join(data_dir, "ORN_ROAD_NET_ELEMENT.shp"))
    # Load road attributes (all with delimiter=';')
    speed = pd.read_csv(os.path.join(data_dir, "ORN_SPEED_LIMIT.csv"), delimiter=";")
    lanes = pd.read_csv(
        os.path.join(data_dir, "ORN_NUMBER_OF_LANES.csv"), delimiter=";"
    )
    road_class = pd.read_csv(
        os.path.join(data_dir, "ORN_ROAD_CLASS.csv"), delimiter=";"
    )
    # Load crash data
    ksi = pd.read_csv(os.path.join(base, "data/KSI.csv"))
    return roads, speed, lanes, road_class, ksi


if __name__ == "__main__":
    roads, speed, lanes, road_class, ksi = load_data()
    print("Road segments:", len(roads))
    print("Speed records:", len(speed))
    print("Lanes records:", len(lanes))
    print("Road class records:", len(road_class))
    print("Crashes:", len(ksi))
    print("\nColumns in roads:", list(roads.columns))
    print("Columns in speed:", list(speed.columns))
    print("Columns in lanes:", list(lanes.columns))
    print("Columns in road_class:", list(road_class.columns))
    print("Columns in ksi:", list(ksi.columns))
