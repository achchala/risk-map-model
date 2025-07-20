# Toronto Road Segment Crash Risk Prediction MVP

## Project Overview
This project builds a geospatial machine learning model to predict crash risk levels for road segments in Toronto using open traffic collision data. The model outputs risk labels (Low/Medium/High) and scores per road segment, suitable for visualization as heatmaps or geo-layers.

## Project Structure
```
risk-map-model/
├── data/                          # Raw datasets
│   ├── Traffic_Collisions_Open_Data_2437597425626428496.xlsx
│   ├── TOTAL_KSI_6386614326836635957.csv
│   └── Centreline - Version 2 - 4326.geojson
├── src/                           # Source code
│   ├── data_processing/           # Data cleaning and preprocessing
│   ├── feature_engineering/       # Feature creation and engineering
│   ├── modeling/                  # ML model training and evaluation
│   └── visualization/             # Mapping and plotting utilities
├── models/                        # Trained model files
├── outputs/                       # Generated outputs
│   ├── maps/                      # Interactive maps and visualizations
│   ├── models/                    # Saved model artifacts
│   └── reports/                   # Analysis reports and metrics
├── notebooks/                     # Jupyter notebooks for exploration
└── docs/                          # Documentation
```

## Data Sources
1. **Traffic Collision Data (MVC)**: Toronto Police Open Data - general collision records
2. **Killed or Seriously Injured (KSI) Data**: Toronto Police Open Data - severe crash cases
3. **Road Network Geometry**: Toronto Open Data Portal - road segment geometries

## Key Features
- **Spatial Processing**: 20m buffer spatial join for crash-to-segment matching
- **Feature Engineering**: Crash counts, KSI ratios, temporal patterns, road characteristics
- **Risk Classification**: 3-class model (Low/Medium/High) with probability scores
- **Geospatial Output**: GeoJSON format for mapping applications

## Model Approach
- **Algorithm**: Random Forest Classifier (interpretable, handles mixed data types)
- **Evaluation**: 5-fold cross-validation with class balancing
- **Features**: Segment-level crash statistics, temporal patterns, road class
- **Output**: Risk labels + probability scores per road segment

## Installation
```bash
pip install -r requirements.txt
```

## Usage
1. Place your datasets in the `data/` folder
2. Run the main pipeline: `python src/main.py`
3. View outputs in `outputs/` folder

## Limitations (MVP)
- No traffic volume/exposure data
- Static analysis (no temporal filtering)
- No weather integration
- Based on crash frequency, not crash rate per vehicle

## Future Enhancements
- Real-time updates
- Weather integration
- Traffic volume data
- Temporal risk patterns
- API deployment 