# backend API

flask api server for serving risk predictions to the iOS app

## Setup

1. install dependencies:
```bash
pip install -r requirements.txt
```

2. ensure that the trained model exists at:
```
../outputs/models/toronto_risk_model.joblib
```

3. run the server:
```bash
python app.py
```

The API will be available (for local testing) at `http://localhost:8000`

## Endpoints

### GET `/api/health`
health check endpoint

### POST `/api/risk-predictions`
get risk predictions for a geographic region

**Request:**
```json
{
  "north": 43.7,
  "south": 43.6,
  "east": -79.3,
  "west": -79.4
}
```

**Response:**
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

### POST `/api/risk-prediction`
get risk prediction for a specific location

**Request:**
```json
{
  "latitude": 43.6532,
  "longitude": -79.3832
}
```

**Response:**
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

## Features

- model integration: uses the trained Random Forest model from the pipeline
- pre-processed data: fast responses using pre-computed predictions
- on-demand predictions: Ccn generate fresh predictions using the model
- probability scores: returns prediction probabilities (not just labels)
- CORS enabled for iOS app integration

## How It Works

1. **pre-processed data path**:
   - loads GeoJSON with pre-computed predictions from the pipeline
   - returns predictions instantly for visible map regions
   - optionally refreshes predictions using the model if available

2. **on-demand path**:
   - falls back to raw road network if pre-processed data unavailable
   - returns basic segment info (full prediction requires crash data)

3. **model predictions**:
   - uses `ModelTrainer.predict()` and `predict_proba()` for fresh predictions
   - handles feature extraction and scaling automatically
   - returns both risk label and confidence probabilities

## next steps!!

- [ ] add caching for performance
- [ ] implement proper nearest segment search optimization
- [ ] add authentication/rate limiting
- [ ] deploy to production server
- [ ] add batch prediction endpoint for multiple segments

