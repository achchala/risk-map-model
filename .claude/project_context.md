# Risk Map Model - Project Context

## Project Overview

The **Risk-Map-Model** is a full-stack geospatial machine learning application that predicts and visualizes crash risk levels for road segments in Toronto. The system helps users make informed decisions about route selection by providing transparency about road safety based on historical crash data.

### Primary Value Proposition
Transform Toronto's historical crash data (618K+ collisions) into real-time safety guidance through ML-powered risk predictions and safer route recommendations.

---

## System Architecture

### Components

1. **ML Pipeline** (Python - `/src/`)
   - Offline data processing and model training
   - Random Forest classifier with SMOTE for class balancing
   - Processes 618K collision records + 18K KSI crashes + 65K road segments
   - Outputs: trained model (joblib) + pre-processed GeoJSON

2. **Backend API** (Flask - `/backend-api/`)
   - REST API serving ML predictions
   - Runs on `http://localhost:8000`
   - Endpoints: `/api/health`, `/api/risk-predictions`, `/api/risk-prediction`
   - Fast response times via pre-computed predictions

3. **iOS Mobile App** (SwiftUI - `/ios-app/RiskMapApp/`)
   - Interactive map visualization
   - Safer route planning with risk comparison
   - MapKit integration for routing
   - Location search with autocomplete

### Data Flow

```
Offline: Raw Data → ML Pipeline → Trained Model + GeoJSON
Online: iOS App → Backend API → Pre-computed Predictions → User
```

---

## Key Technologies

### Backend
- **Python 3.8+**, Flask, Scikit-learn, GeoPandas, Shapely
- **ML Model:** Random Forest (100 trees, max depth 10)
- **Spatial:** R-tree indexing, 20m buffer distance
- **Coordinate System:** EPSG:4326 (WGS84)

### iOS
- **Swift 5.9+**, SwiftUI, MapKit, CoreLocation, Combine
- **Architecture:** MVVM pattern
- **Min iOS:** 17.0+

### Data
- **GeoJSON** for spatial data
- **Joblib** for model serialization
- **JSON** for API communication

---

## Domain Concepts

### Risk Levels
- **High Risk:** >2 KSI crashes OR >10 total crashes
- **Medium Risk:** ≥1 KSI crash OR >5 total crashes
- **Low Risk:** Everything else

### KSI (Killed or Seriously Injured)
Severe crashes resulting in fatalities or serious injuries. Weighted higher in risk calculations.

### Road Segment
A section of road with consistent characteristics (road class, name). Average ~150m length.

### Risk Score
Numerical score used for route comparison:
- High risk = 3.0
- Medium risk = 2.0
- Low risk = 1.0

---

## Project Structure

```
risk-map-model/
├── backend-api/           # Flask REST API
│   ├── app.py            # Main API server (single file)
│   └── requirements.txt
├── ios-app/              # iOS mobile app
│   └── RiskMapApp/
│       ├── Models/       # Data models
│       ├── Services/     # API + route services
│       └── Views/        # SwiftUI screens
├── src/                  # ML pipeline modules
│   ├── data_processing/
│   ├── feature_engineering/
│   ├── models/
│   └── visualization/
├── data/                 # Raw input data
├── outputs/              # Generated artifacts
│   ├── models/          # Trained models
│   ├── reports/         # GeoJSON exports
│   └── maps/            # HTML visualizations
├── config.py            # Centralized configuration
└── run_risk_analysis.py # Main pipeline script
```

---

## Development Setup

### Backend API
```bash
cd backend-api
pip install -r requirements.txt
python app.py  # Runs on http://localhost:8000
```

### iOS App
1. Open `ios-app/RiskMapApp/RiskMapApp.xcodeproj` in Xcode
2. Update backend URL in `RiskService.swift:19` if needed
3. Build and run (⌘R)

### ML Pipeline
```bash
pip install -r requirements.txt
python run_risk_analysis.py
```

---

## Common Patterns

### API Response Format
All API endpoints return JSON with consistent structure:
```json
{
  "id": "unique_id",
  "risk_label": "high" | "medium" | "low",
  "confidence": 0.0-1.0,
  "coordinates": [{"latitude": 43.6, "longitude": -79.3}]
}
```

### SwiftUI State Management
- Use `@Published` properties in `ObservableObject` services
- Views observe services via `@StateObject` or `@ObservedObject`
- Reactive updates via Combine framework

### Error Handling
- Backend: Try-except with logging, return 500/400 errors
- iOS: Result type or throws, display user-friendly alerts

---

## Important Constraints

1. **Toronto-only:** Risk data limited to Toronto area
2. **Historical data:** No real-time crash updates
3. **No authentication:** Backend currently has no auth (dev only)
4. **MapKit routing:** Can't control specific roads, only analyze routes
5. **50m matching threshold:** Route points matched to segments within 50m

---

## Performance Targets

- **API Response:** <50ms (pre-processed), <500ms (model inference)
- **Route Calculation:** 2-5 seconds total
- **Map Load:** 1-2 seconds
- **Max Segments:** 500 per API request
- **Max Route Points:** 100 sample points

---

## Future Roadmap

1. Authentication and rate limiting for backend
2. Multi-city support
3. Real-time traffic integration
4. Turn-by-turn navigation in iOS app
5. Time-of-day specific risk predictions
6. Android app
7. Web viewer

---

## Testing

- **ML Pipeline:** Run `python run_risk_analysis.py` and verify outputs
- **Backend API:** Test with curl or Postman
- **iOS App:** Manual testing on simulator or device
- No automated test suites currently exist

---

## Deployment Status

**Current:** Development only (localhost)
**Production:** Not ready - requires authentication, HTTPS, load balancing, monitoring

---

## Git Workflow

- **Main branch:** `main`
- **Current branch:** `adriel-ui-branch`
- Recent work: Search locations, routing, comparison cards, debug mode

---

## Contact & Documentation

- **README:** See `/README.md` for ML pipeline details
- **API Docs:** See `/backend-api/README.md`
- **Integration:** See `/backend-api/INTEGRATION.md`
