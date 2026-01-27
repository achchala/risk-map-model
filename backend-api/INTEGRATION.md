# Backend API Integration Guide

## Overview

the backend API (`backend-api/app.py`) is integrated with the pre-existing ML pipeline. It uses:
- the trained model from `outputs/models/toronto_risk_model.joblib`
- the existing `ModelTrainer` class
- data processing pipeline modules
- pre-processed data from the pipeline output

## Architecture

```
iOS App
  ↓ HTTP Requests
Backend API (Flask)
  ↓
  ├─→ Pre-processed Data (Fast Path)
  │   └─→ outputs/reports/toronto_road_risk.geojson
  │
  └─→ Model Predictions (On-Demand)
      ├─→ ModelTrainer.load_model()
      ├─→ ModelTrainer.predict()
      └─→ ModelTrainer.model.predict_proba()
```

## Data Flow

### 1. pre-processed data path 

first, you run the full pipeline (`run_risk_analysis.py`), and it generates:
- `outputs/reports/toronto_road_risk.geojson`, which result in pre-computed predictions for all segments

The API loads this file and filters segments by bounding box (map region), returns predictions instantly, and optionally refreshes predictions using the model if features are available

### 2. model prediction path

if pre-processed data isn't available or the user wants fresh predictions:
- loads the trained model using `ModelTrainer.load_model()`
- extracts features from segment data
- uses `model.predict()` and `model.predict_proba()` for predictions
- returns risk label + confidence probabilities

## Key Functions

### `_predict_segment_risk(segment, model_trainer)`

1. extracts features from a segment row
2. ensures all required feature columns are present
3. scales features using the model's scaler
4. gets prediction and probabilities from the model
5. decodes the prediction back to label ('low', 'medium', 'high')
6. Returns probabilities for all three classes

### Model Integration

the API uses the existing `ModelTrainer` class
- `model_trainer.model` - RandomForestClassifier
- `model_trainer.scaler` - StandardScaler for feature scaling
- `model_trainer.label_encoder` - maps labels to/from integers
- `model_trainer.feature_columns` - list of features the model expects

## Setup

1. **run the full pipeline** to generate pre-processed data:
   ```bash
   python run_risk_analysis.py
   ```
   this creates `outputs/reports/toronto_road_risk.geojson`

2. **start the API server**:
   ```bash
   cd backend-api
   pip install -r requirements.txt
   python app.py
   ```

3. **the API will**:
   - load the model from `outputs/models/toronto_risk_model.joblib`
   - load pre-processed data from `outputs/reports/toronto_road_risk.geojson`
   - load raw road network as fallback

## API Endpoints

### `GET /api/health`
returns status of:
- whether the odel loaded
- whether the pre-processed data loaded
- whether the road network loaded
- the umber of pre-processed segments

### `POST /api/risk-predictions`
**Input**: bounding box (north, south, east, west)
**Output**: array of road segments with risk predictions

**Uses**:
- pre-processed data if available (fastest)
- model predictions if features available (accurate)
- falls back to basic info if neither available

### `POST /api/risk-prediction`
**input**: single location (latitude, longitude)
**output**: risk prediction with probabilities

**uses**:
- finds nearest segment from pre-processed data
- gets model prediction with probabilities
- returns detailed segment info

## model prediction details

when the model makes a prediction:

1. **feature extraction**: extracts all features the model expects
   - road characteristics (road_class_*, segment_length)

2. **scaling**: features are scaled using the same StandardScaler from training

3. **prediction**: 
   - `model.predict()` → risk label (0=low, 1=medium, 2=high)
   - `model.predict_proba()` → probabilities [low, medium, high]

4. **decoding**: integer prediction is decoded back to string label

## performance

- **pre-processed data**: ~10-50ms per request (very fast)
- **Mmdel prediction**: ~100-500ms per segment (slower but accurate)
- **fallback**: ~50-200ms (basic info only)

## Troubleshooting

### model not loading?
- check that `outputs/models/toronto_risk_model.joblib` exists
- run the training pipeline first

### pre-processed data not found?
- run `python run_risk_analysis.py` to generate it
- or the API will use model predictions (slower)





