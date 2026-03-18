# backend API

flask api server for serving risk predictions to the iOS app

## Setup

1. install dependencies:
```bash
pip install -r requirements.txt
```

2. ensure the trained model and panel exist:
```
../outputs/models/toronto_temporal_count_model.pkl
../outputs/reports/panel_latest.parquet
```
Run `python src/models/hurdle_model/train_temporal_model.py` from project root to generate them.

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

### GET `/api/risk-definition`
Returns p70, p90 thresholds, risk label descriptions, and optional `featureImportance` for ranking risk drivers.

### POST `/api/routes/safety-aware`
Returns fastest and safer route options between origin and destination. Request body: `{ "origin": { "latitude", "longitude" }, "destination": { "latitude", "longitude" }, "beta": 0.1 }`.

### POST `/api/risk-prediction`
Get risk prediction for a specific location

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

- **Temporal hurdle model**: Uses `HurdleTemporalTrainer` (two-stage: P(crash) × E[count|crash])
- **Panel-based inference**: Loads `panel_latest.parquet` and predicts λ per segment for latest window
- **Risk drivers**: Returns contributing factors (traffic, weather, history) ranked by feature importance
- **Safety-aware routing**: `/api/routes/safety-aware` for fastest vs safer route options
- **CORS enabled** for iOS app integration

## How It Works

1. **Startup**: Loads road network, panel data, and trained model; computes λ map for latest window
2. **risk-predictions**: Filters segments by bbox, returns risk labels + drivers + explanation
3. **routes/safety-aware**: Dijkstra on road graph with risk-weighted edge costs; returns two route options

## next steps!!

- [ ] add caching for performance
- [ ] implement proper nearest segment search optimization
- [ ] add authentication/rate limiting
- [ ] deploy to production server
- [ ] add batch prediction endpoint for multiple segments

