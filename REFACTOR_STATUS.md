# Predictive Crash Model Refactor — Status vs Plan

This document tracks what has been done from the original refactor plan, what remains, and how to check and test it.

---

## Plan sections (A–K) — Done vs remaining

| Section | Description | Status | Notes |
|--------|-------------|--------|--------|
| **A** | Predictive target (λ crash rate, conversion to P and route risk) | **Done** | Defined and used in `TemporalCountModelTrainer` and `road_graph` |
| **B** | Prediction horizon H, steps_ahead = H/W, sparsity analysis | **Partial** | PanelConfig enforces H/W; no dedicated sparsity script; training uses 24h windows for memory |
| **C** | Event-level crash assignments (CENTRELINE_ID, event_datetime, schema) | **Done** | `perform_spatial_join_event_level()` in `spatial_join_fast.py` |
| **D** | Temporal panel (no leakage, past-only lags, weather join) | **Done** | `panel_builder.py`; weather join is optional (no live weather yet) |
| **E** | Temporal train/val/test split and evaluation | **Done** | `temporal_train_val_test_split()`, used in `TemporalCountModelTrainer` |
| **F** | Count model (Poisson-style GBDT), λ → routing | **Done** | `TemporalCountModelTrainer` + `road_graph` risk-to-cost |
| **G** | Calibration (λ → P, isotonic, save/load) | **Done** | In `TemporalCountModelTrainer` |
| **H** | Routing graph, fastest/safer routes, snap-to-graph, maps export | **Done** | `road_graph.py`, `maps_export.py`; backend `/api/routes/safety-aware` |
| **I** | CRS/distance verification | **Done** | `verify_crs_and_distance()` in `spatial_join_fast.py` |
| **J** | Acceptance criteria | **Done** | Encoded in `tests/test_acceptance.py` |
| **K** | Deliverables (files to add/change) | **Done** | All listed files created or updated |

---

## What’s done (summary)

- **Data**
  - Event-level spatial join with stable `segment_id` = CENTRELINE_ID and correct DATE+HOUR.
  - CRS check so buffers are in meters.
  - Panel builder: segment×window panel, past-only lags, future label with steps_ahead, optional weather columns.
  - Panel restricted to “active” segments (with at least one crash) to avoid OOM.

- **Model**
  - `TemporalCountModelTrainer`: Poisson GBDT, temporal split, calibration, save/load (model + calibrator).
  - Training script: `train_temporal_model.py` (uses 24h windows to keep panel size manageable).

- **Routing**
  - Directed graph from road network, travel time by road class, risk-to-cost (λ·t, beta), fastest vs safer routes, route risk summary (expected crashes, route probability).
  - Node geometry from segment endpoints, snap-to-graph with max distance.
  - Maps export: simplify + cap waypoints (e.g. 25).

- **Backend**
  - Loads temporal model + panel snapshot; `/api/routes/safety-aware` returns fastest route, safer route, time/risk summaries, avoided segments, and risk drivers.

- **Tests**
  - `tests/test_acceptance.py`: no leakage (steps_ahead), temporal split ordering, inference on unseen windows, stable segment IDs, routing math.

---

## What still needs to be done

1. **Run training successfully**
   - After the OOM fix (24h windows + active segments only), run:
     - `python train_temporal_model.py`
   - Confirm it finishes and produces:
     - `outputs/models/toronto_temporal_count_model.pkl`
     - `outputs/reports/panel_latest.parquet`

2. **Sparsity analysis (plan Section B)**
   - Add a small script or notebook that, for H ∈ {1h, 6h, 24h}, computes zero rate and basic stats on the panel and (optionally) recommends H or ZIP/ZINB. Not required for the current 24h training path.

3. **Weather**
   - Panel and config support weather (grid join, cache path). Still needed: a real weather source and (if desired) a job to fill the cache / panel weather columns.

4. **Optional: `run_risk_analysis.py`**
   - Plan said to update the main pipeline to use event-level → panel → temporal model. Currently it still runs the old classifier pipeline (aggregate join, rule-based labels, Random Forest). You can keep the old pipeline for legacy outputs and use `train_temporal_model.py` + backend for the new predictive flow.

5. **Optional: 20m distance test**
   - Plan Section I mentioned a test that a 20m distance check behaves as expected. You can add a small test (e.g. in `tests/`) that uses projected CRS and asserts ~20 m for two points.

---

## How to check and test

### 1. Acceptance tests (new pipeline logic)

From project root:

```bash
python -m pytest tests/test_acceptance.py -v
```

- **No leakage:** `future_crash_count` is shifted by `steps_ahead`; current-window counts are not the target.
- **Temporal split:** Test windows are strictly after train windows.
- **Inference on unseen:** Model predicts on shifted “future” windows.
- **Stable IDs:** `segment_id` matches CENTRELINE_ID; graph edges use it.
- **Routing math:** Route expected crashes = Σ(λᵢ·tᵢ); route probability = 1 − exp(−Σ(λᵢ·tᵢ)).

If any test fails, fix the corresponding part of the pipeline (panel builder, trainer, or routing).

### 2. Train temporal model and check outputs

```bash
python train_temporal_model.py
```

- Expect: no crash; logs show panel shape, train/val/test window counts, and MAE/RMSE/Poisson deviance.
- Check files exist:
  - `outputs/models/toronto_temporal_count_model.pkl`
  - `outputs/reports/panel_latest.parquet`

### 3. Backend health and safety-aware routing

Start the API (from project root, with model + panel from step 2):

```bash
cd backend-api
pip install -r requirements.txt   # if needed
python app.py
```

Then:

- **Health:**  
  `GET http://localhost:8000/api/health`  
  - Expect: `temporal_model_loaded`, `routing_graph_built`, `panel_loaded` true when model and panel are present.

- **Safety-aware routes:**  
  `POST http://localhost:8000/api/routes/safety-aware`  
  Body (JSON):
  ```json
  {
    "origin": {"latitude": 43.65, "longitude": -79.38},
    "destination": {"latitude": 43.70, "longitude": -79.40},
    "beta": 0.1
  }
  ```
  - Expect: JSON with `fastest` and `safer` (nodes, segmentIds, summary with totalTravelTimeHours, expectedCrashes, routeProbability), `avoidedSegments` (with segmentId, lambdaPerHour, riskDrivers), and `betaHoursPerExpectedCrash`.

### 4. CRS / distance (Section I)

- In code: `verify_crs_and_distance(road_network)` is called at the start of the spatial join. If buffers were in degrees, it would raise.
- Optional: add a unit test that creates two points ~20 m apart in projected CRS and asserts distance ≈ 20 m.

### 5. Old pipeline (optional)

To confirm the legacy flow still runs (no refactor breakage):

```bash
python run_risk_analysis.py
```

- Uses aggregate join, rule-based labels, and Random Forest; produces maps/reports and old GeoJSON. It does **not** use the new temporal model or panel.

---

## Quick checklist

- [ ] `python -m pytest tests/test_acceptance.py -v` passes
- [ ] `python train_temporal_model.py` completes and creates `.pkl` and `panel_latest.parquet`
- [ ] Backend starts; `/api/health` shows temporal model and panel loaded
- [ ] `POST /api/routes/safety-aware` with origin/destination returns fastest + safer routes and risk summaries
- [ ] (Optional) Sparsity analysis script for H; weather source + cache
- [ ] (Optional) Update `run_risk_analysis.py` to optionally use the new pipeline
- [ ] (Optional) Add explicit 20 m distance test

This status and the steps above give you a direct way to verify what’s done from the plan and what to run to check and test it.
