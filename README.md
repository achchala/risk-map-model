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
│   ├── data_processing/           # Data cleaning and spatial joins
│   ├── feature_engineering/       # Feature creation and risk labeling
│   ├── models/                    # Model training and evaluation
│   └── visualization/             # Interactive maps and dashboards
├── outputs/                       # Generated outputs
│   ├── maps/                      # Interactive risk maps (HTML)
│   ├── models/                    # Trained model artifacts
│   └── reports/                   # Analysis reports and data exports
├── run_risk_analysis.py           # Main pipeline script
├── config.py                      # Configuration settings
└── requirements.txt               # Python dependencies
```

## Data Sources
1. **Traffic Collision Data (MVC)**: Toronto Police Open Data - general collision records
2. **Killed or Seriously Injured (KSI) Data**: Toronto Police Open Data - severe crash cases
3. **Road Network Geometry**: Toronto Open Data Portal - road segment geometries

## Key Features
- **Spatial Processing**: Optimized spatial join for crash-to-segment matching
- **Feature Engineering**: Crash statistics, temporal patterns, road characteristics
- **Risk Classification**: 3-class model (Low/Medium/High) with confidence scores
- **Interactive Visualization**: HTML dashboards with risk maps and performance metrics
- **Model Performance**: Detailed evaluation including confusion matrix and per-class metrics
- **Data Export**: GeoJSON and CSV formats for further analysis

## Model Approach
- **Algorithm**: Random Forest Classifier with SMOTE for class balancing
- **Evaluation**: Cross-validation with detailed performance metrics (accuracy, precision, recall, F1-score)
- **Features**: Segment-level crash statistics, temporal patterns, road characteristics
- **Output**: Risk labels + confidence scores per road segment
- **Performance Reporting**: Confusion matrix and per-class performance analysis

## Installation
```bash
pip install -r requirements.txt
```

## Usage
1. Place your datasets in the `data/` folder
2. Run the main pipeline: `python run_risk_analysis.py`
3. View outputs in `outputs/` folder:
   - **Interactive Dashboard**: `outputs/maps/toronto_risk_analysis_dashboard.html`
   - **Risk Map**: `outputs/maps/toronto_risk_map.html`
   - **Summary Report**: `outputs/reports/risk_analysis_summary.html`
   - **Data Exports**: `outputs/reports/` (CSV and GeoJSON files)

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