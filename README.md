# Toronto Road Risk Prediction & Visualization

## Project Overview
This project predicts and visualizes crash risk scores for road segments in Toronto, using historical crash data and detailed road network information. The goal is to provide actionable insights for city planners, self-driving car developers, and logistics companies.

## Folder Structure
```
implementation-1/
  data/
    KSI.csv, KSI.geojson         # Crash data
    ORN_data/                    # Road network and attributes
      ORN_ROAD_NET_ELEMENT.shp   # Road segment geometries
      ORN_ROAD_NET_ELEMENT.dbf   # Attributes for segments
      ...                        # Other road attributes (CSV)
  notebooks/
    exploratory_analysis.ipynb   # EDA and prototyping
  scripts/
    data_loading.py              # Load and join data
    feature_engineering.py       # Feature extraction
    modeling.py                  # Model training and prediction
  .gitignore
  README.md
```

## Data Sources
- **KSI.csv / KSI.geojson**: Killed or Seriously Injured collision data
- **ORN_data**: Ontario Road Network shapefiles and attributes

## Workflow
1. **Data Loading**: Load road network and crash data
2. **Feature Engineering**: Extract features (speed, lanes, class, weather, time, etc.)
3. **Modeling**: Train a model to predict crash risk per segment
4. **Visualization**: Display risk scores on a map, with filtering by time/weather
5. **UI**: Build an interactive web app for users

## Setup Instructions
1. Clone the repo
2. Install dependencies (see below)
3. Place data in `implementation-1/data/` as shown above
4. Run notebooks/scripts for EDA, modeling, and visualization
5. Launch the UI with `python ui/app.py`

## Dependencies
- Python 3.9+
- geopandas, pandas, shapely, scikit-learn, folium/streamlit/dash (for UI)

## Usage
- See `notebooks/exploratory_analysis.ipynb` for EDA and prototyping
- Use scripts in `scripts/` for data processing and modeling
- Launch the UI to interact with the risk map

## Authors
- Achchala, Adriel, Anastan

---
For more details, see the code and notebooks in each folder. 