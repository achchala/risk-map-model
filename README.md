# Toronto Road Segment Crash Risk Prediction

A geospatial ML system that predicts crash risk (λ) for road segments in Toronto and powers an iOS app with safety-aware routing.

## Overview

- **Model**: Two-stage hurdle model — Stage 1 predicts P(crash), Stage 2 predicts E[count | crash]. Combined: λ = P × E.
- **Output**: Predicted crash rate λ per segment-hour; risk labels (Low/Medium/High) from percentile thresholds.
- **Stack**: Python (scikit-learn, GeoPandas), Flask API, SwiftUI iOS app with MapKit.

## Project Structure

```
risk-map-model/
├── data/                          # Raw datasets
│   ├── Traffic_Collisions_Open_Data_*.xlsx
│   ├── TOTAL_KSI_*.csv
│   ├── Centreline - Version 2 - 4326.geojson
│   ├── model_dataset.csv          # ADT/speed
│   ├── historicalweather.csv
│   ├── tmc_raw_data_*.csv         # TMC intersection counts
│   ├── School locations-*.csv
│   └── TTC Routes and Schedules Data/  # Optional GTFS
├── src/
│   ├── data_processing/           # Data loading, spatial joins
│   ├── feature_engineering/      # Panel builder, temporal features
│   ├── models/                    # Model training
│   │   ├── hurdle_model/          # train_temporal_model.py (main entry)
│   │   ├── hurdle/               # HurdleTemporalTrainer
│   │   ├── temporal_count/       # Single-stage Poisson
│   │   └── legacy_classifier/    # Legacy Random Forest
│   ├── routing/                   # Road graph, Dijkstra, safety-aware routes
│   └── visualization/             # Risk maps
├── outputs/
│   ├── models/                    # toronto_temporal_count_model.pkl
│   └── reports/                   # panel_latest.parquet, diagnostics
├── backend-api/                   # Flask API for iOS
├── ios-app/                       # SwiftUI iOS app
├── docs/                          # Architecture, routing, formulas
├── config.py
└── requirements.txt
```

## Data Sources


| Source                   | Purpose                                            |
| ------------------------ | -------------------------------------------------- |
| Traffic Collisions (MVC) | Crash events for spatial join                      |
| KSI Data                 | Severe crash events                                |
| Centreline GeoJSON       | Road network geometry                              |
| model_dataset.csv        | ADT, speed, exposure                               |
| historicalweather.csv    | Temperature, precipitation, snow                   |
| TMC data                 | Pedestrian/cyclist/vehicle counts at intersections |
| School locations         | School zone flag                                   |
| TTC GTFS (optional)      | Transit frequency                                  |


## Model

- **Algorithm**: Hurdle model — HistGradientBoostingClassifier (Stage 1) + HistGradientBoostingRegressor with Poisson loss (Stage 2).
- **Features**: Traffic volume, road geometry, weather, temporal indicators, crash lags, historical profiles, TMC exposure, school/transit context.
- **Risk labels**: Low ≤ p70, Medium p70–p90, High > p90 (percentiles of λ).

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1. Train the model

```bash
python src/models/hurdle_model/train_temporal_model.py
```

Outputs:

- `outputs/models/toronto_temporal_count_model.pkl`
- `outputs/reports/panel_latest.parquet`

### 2. Run the backend API

```bash
cd backend-api
pip install -r requirements.txt
python app.py
```

API runs at `http://localhost:8000`. See `backend-api/README.md` for endpoints.

### 3. Run the iOS app

```bash
cd ios-app/RiskMapApp
open RiskMapApp.xcodeproj
```

- **Simulator**: Uses `localhost:8000`
- **Device**: Update `baseURL` in `RiskService.swift` to your Mac's IP (same Wi‑Fi)

See `ios-app/README.md` and `ios-app/QUICK_START.md` for details.

## Key Features

- **Temporal panel**: Segment × window with crash counts, lags, weather, road attributes.
- **Safety-aware routing**: Dijkstra on road graph with risk-weighted edge costs; offers fastest vs safer route.
- **Risk explanation**: UI shows contributing factors (traffic, weather, history, etc.) ranked by model importance.
- **Weather integration**: Historical (NOAA) and live (WeatherAPI.com) for risk adjustment.

## API Endpoints


| Method | Endpoint                   | Purpose                       |
| ------ | -------------------------- | ----------------------------- |
| POST   | `/api/risk-predictions`    | Risk predictions for map bbox |
| GET    | `/api/risk-prediction`     | Risk for a single location    |
| GET    | `/api/risk-definition`     | p70, p90, feature importance  |
| POST   | `/api/routes/safety-aware` | Fastest + safer route options |




