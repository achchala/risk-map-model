iOS app for visualizing road crash risk predictions in Toronto

## requirements

- iOS 17.0+
- Xcode 15.0+
- Swift 5.9+

## setup instructions!!

### 1. open the project

```bash
cd ios-app/RiskMapApp
open RiskMapApp.xcodeproj
```

### 2. configure API endpoint

update the `baseURL` in `RiskService.swift`:

```swift
private let baseURL = "https://your-api-domain.com/api"
```

### 3. backend API requirements

the app expects a REST API with the following endpoints:

#### GET/POST `/api/risk-predictions`
returns risk predictions for a geographic region.

**request body:**
```json
{
  "north": 43.7,
  "south": 43.6,
  "east": -79.3,
  "west": -79.4
}
```

**response:**
```json
[
  {
    "id": "segment_123",
    "LINEAR_NAME": "Yonge Street",
    "ROAD_CLASS": "Arterial",
    "segment_length": 150.5,
    "risk_label": "high",
    "confidence": 0.85,
    "num_total_crashes": 15,
    "num_ksi_crashes": 3,
    "fatality_count": 1,
    "coordinates": [
      {"latitude": 43.6532, "longitude": -79.3832}
    ]
  }
]
```

#### POST `/api/risk-prediction`
returns risk prediction for a specific location.

**request Body:**
```json
{
  "latitude": 43.6532,
  "longitude": -79.3832
}
```

**response:**
```json
{
  "riskLevel": "high",
  "confidence": 0.85,
  "probabilities": {
    "low": 0.10,
    "medium": 0.05,
    "high": 0.85
  },
  "segmentInfo": { ... }
}
```

### 4. build and run

1. select a simulator or connected device
2. press `Cmd + R` to build and run
3. grant location permissions when prompted

## project structure

```
RiskMapApp/
├── RiskMapApp.swift          # app entry point
├── ContentView.swift         # main tab view
├── Models/
│   └── RiskModels.swift      # data models
├── Services/
│   └── RiskService.swift     # API service
├── Views/
│   ├── MapView.swift         # map with risk annotations
│   ├── RiskDetailView.swift  # segment detail view
│   ├── RiskListView.swift    # high-risk roads list
│   └── SettingsView.swift    # settings screen
└── Info.plist                # app config
```

## Backend Integration

to connect this app to your Python backend:

1. **create a Flask/FastAPI backend** that serves the trained model
2. **expose REST endpoints** matching the API specification above
3. **load the trained model** from `outputs/models/toronto_risk_model.joblib`
4. **return predictions** in the expected JSON format







