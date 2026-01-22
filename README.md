# MGTE Capstone 2026

## Project Overview

ML model to predict crash risk levels for road segments in Toronto using open traffic collision data. The model outputs risk labels (Low/Medium/High) and confidence scores per road segment, suitable for visualization as heatmaps or geo-layers.

The project consists of three main components:
1. **ML Pipeline**: Python-based data processing, feature engineering, and model training
2. **Backend API**: Flask REST API server that serves predictions to the iOS app
3. **iOS App**: SwiftUI mobile application for visualizing risk predictions on an interactive map

## Project Structure

```
risk-map-model/
├── data/                          # Raw datasets (not in git)
│   ├── Traffic_Collisions_Open_Data_*.xlsx
│   ├── TOTAL_KSI_*.csv
│   └── Centreline - Version 2 - 4326.geojson
├── src/                           # Source code
│   ├── data_processing/           # Data cleaning and spatial joins
│   │   ├── data_loader.py
│   │   └── spatial_join_fast.py
│   ├── feature_engineering/       # Feature creation and risk labeling
│   │   ├── feature_creator.py
│   │   └── label_generator.py
│   ├── models/                    # Model training and evaluation
│   │   ├── model_trainer.py
│   │   └── model_evaluator.py
│   └── visualization/             # Interactive maps and dashboards
│       └── risk_mapper.py
├── outputs/                       # Generated outputs (not in git)
│   ├── maps/                      # Interactive risk maps (HTML)
│   ├── models/                    # Trained model artifacts
│   │   └── toronto_risk_model.joblib
│   └── reports/                   # Analysis reports and data exports
│       └── toronto_road_risk.geojson
├── ios-app/                       # iOS mobile application
│   ├── RiskMapApp/                # SwiftUI iOS application
│   │   ├── RiskMapApp.swift       # App entry point
│   │   ├── ContentView.swift      # Main tab view
│   │   ├── Models/
│   │   │   └── RiskModels.swift   # Data models
│   │   ├── Services/
│   │   │   ├── RiskService.swift  # API service
│   │   │   └── RouteService.swift # Route planning service
│   │   └── Views/
│   │       ├── MapView.swift      # Map with risk annotations
│   │       ├── RiskDetailView.swift
│   │       ├── RiskListView.swift
│   │       └── SettingsView.swift
│   ├── README.md
│   └── QUICK_START.md
├── backend-api/                   # Flask API server
│   ├── app.py                     # API endpoints
│   ├── requirements.txt           # Python dependencies
│   ├── README.md
│   ├── QUICK_SETUP.md            # Google Maps API setup
│   └── INTEGRATION.md            # Integration details
├── run_risk_analysis.py           # Main pipeline script
├── config.py                      # Configuration settings
├── requirements.txt               # Python dependencies
├── README.md                      # This file
└── TECHNICAL_OVERVIEW.md          # Detailed technical documentation
```

## Installation

### Prerequisites
- Python 3.8+
- iOS 17.0+ (for iOS app)
- Xcode 15.0+ (for iOS app)

## Usage

### 1. Run ML Pipeline

First, place your datasets in the `data/` folder, then run:

```bash
python run_risk_analysis.py
```

### 2. Start Backend API

```bash
cd backend-api
python app.py
```

### 3. Run iOS App

1. **Open in Xcode:**
   ```bash
   cd ios-app/RiskMapApp
   open RiskMapApp.xcodeproj
   ```

2. **Configure API endpoint** in `RiskMapApp/Services/RiskService.swift`:
   ```swift
   private let baseURL = "http://localhost:8000/api" 
   ```

3. **Set up signing**:
   - Select the RiskMapApp target
   - Go to Signing & Capabilities
   - Select your development team

4. **Build and run** (⌘ + R)


## Documentation

- **Technical Overview**: See `TECHNICAL_OVERVIEW.md` for detailed technical documentation
- **Backend API**: See `backend-api/README.md` and `backend-api/INTEGRATION.md`
- **iOS App**: See `ios-app/README.md` and `ios-app/QUICK_START.md`
- **Google Maps Setup**: See `backend-api/QUICK_SETUP.md`