---
name: Predictive Crash Model Refactor Plan
overview: Transform the current rule-based crash risk labeling system into a real predictive ML model for safety-aware routing, addressing data leakage, temporal evaluation, event-level data processing, panel dataset construction, routing integration, and weather features.
todos:
  - id: event-level-joins
    content: Implement event-level crash assignments in spatial_join_fast.py with CENTRELINE_ID as stable segment_id and correct DATE+HOUR combination
    status: completed
  - id: panel-builder
    content: Create panel_builder.py module with vectorized operations, past-only lag features, correct steps_ahead calculation, and weather join
    status: completed
  - id: temporal-splits
    content: Replace random train/test split with temporal split based on unique ordered window_start values
    status: completed
  - id: count-regression
    content: Refactor model from RandomForestClassifier to Negative Binomial/GBDT count regression model
    status: completed
  - id: routing-graph
    content: Build routing graph with NetworkX, implement Dijkstra for fastest/safer routes, and risk-to-cost conversion
    status: completed
  - id: snap-to-graph
    content: Implement snap-to-graph functionality with node geometry construction and max distance handling
    status: completed
  - id: maps-export
    content: Implement maps export with Douglas-Peucker simplification and 25-waypoint cap
    status: completed
  - id: calibration
    content: Add calibration support for converting λ predictions to traversal probabilities with isotonic regression
    status: completed
  - id: weather-integration
    content: Integrate weather features with caching, missingness handling, and grid-based spatial join
    status: completed
  - id: acceptance-tests
    content: "Create acceptance tests for all criteria: no leakage, temporal evaluation, inference on unseen data, stable IDs, routing math"
    status: completed
isProject: false
---

# Predictive Crash-Likelihood Model Refactor Plan

## Executive Summary

This plan transforms the current rule-based crash risk labeling system into a real predictive machine learning model that enables safety-aware routing in a mobile app. The refactor addresses critical issues: data leakage, non-predictive targets, random train/test splits, missing routing infrastructure, unstable segment IDs, and lack of weather integration.

## Current State Analysis

### Confirmed Problems

1. **Non-predictive model**: Predicts rule-based labels (low/medium/high) derived from crash count thresholds
2. **Data leakage**: Features derived from crash totals + simulated temporal features
3. **Random train/test split**: Not temporal, allows future information leakage
4. **Inference on training data**: Model scores the same data used to train
5. **No routing graph**: No edges/nodes, no A*/Dijkstra, no risk-to-cost conversion
6. **Unstable segment_id**: CENTRELINE_ID exists but not used as stable identifier
7. **Weather not implemented**: No weather features in model
8. **Simulated temporal features**: Not using actual DATE+HOUR from crash data
9. **Aggregate-only joins**: No event-level crash assignments
10. **No panel dataset**: No temporal windows for time-series modeling

### Current Architecture

- **Data Processing**: `src/data_processing/spatial_join_fast.py` - aggregates crashes to segments only
- **Feature Engineering**: `src/feature_engineering/feature_creator.py` - creates simulated temporal features
- **Labeling**: `src/feature_engineering/label_generator.py` - rule-based thresholds
- **Model Training**: `src/models/model_trainer.py` - Random Forest classifier on labels
- **Evaluation**: `src/models/model_evaluator.py` - random split evaluation
- **Backend API**: `backend-api/app.py` - serves predictions, no routing

---

## Section A: Define the Predictive Target (Crash Likelihood) for Live Routing

### Target Formulation Options

**Option 1: Crash Rate (λ) per Hour per Segment** (RECOMMENDED)

- **Definition**: λ = expected crashes per hour on segment under conditions (time + weather)
- **Units**: crashes/hour
- **Interpretation**: Poisson rate parameter for crash occurrence
- **Conversion to routing cost**: 
  - Traversal probability: P(crash during traversal) = 1 - exp(-λ · t) where t = travel_time_hours
  - Route expected crashes: Σ(λᵢ · tᵢ) for all segments i in route
  - Route probability: P(any crash on route) = 1 - exp(-Σ(λᵢ · tᵢ))
- **Advantages**: 
  - Mathematically sound for count data
  - Directly interpretable as intensity
  - Handles zero crashes naturally
  - Scales with travel time
- **Model**: Negative Binomial regression (handles overdispersion) or Poisson if variance ≈ mean

**Option 2: Crash Probability per Traversal**

- **Definition**: P(crash | traverse segment at time T with weather W)
- **Units**: probability [0, 1]
- **Interpretation**: Bernoulli probability per traversal
- **Conversion to routing cost**: 
  - Route probability: 1 - Π(1 - Pᵢ) for all segments i
  - Route expected crashes: Σ(Pᵢ) (approximation for small P)
- **Advantages**: Direct probability interpretation
- **Disadvantages**: Less natural for count data, requires calibration

### Selected Target: Option 1 (Crash Rate λ)

**Rationale**:

- Crash data is count data (0, 1, 2, ... crashes per hour)
- Poisson/Negative Binomial models are standard for count outcomes
- Rate parameter naturally scales with exposure (travel time)
- Conversion to probability is mathematically rigorous

**Implementation**:

- Target variable: `crash_count_per_hour` = crash_count / exposure_hours
- Exposure = number of hours in time window (e.g., 1h, 6h, 24h)
- Model outputs: λ (crashes/hour) conditioned on features (time, weather, road characteristics)
- Inference: Given current time T and weather W, predict λ(T, W) for each segment

**Edge Cases**:

- Zero crashes: λ = 0 is valid (no crashes expected)
- Long routes: Sum λᵢ·tᵢ across all segments
- Routes longer than prediction horizon: Use λ for first H hours, then fallback to historical average

---

## Section B: Decide Prediction Horizon H (Data-Driven)

### Sparsity Analysis Requirements

**Analysis Steps**:

1. **Load event-level crash data** with DATE + HOUR
2. **Create time windows** of size W hours (e.g., W=1, W=6, W=24)
3. **For each horizon H ∈ {1h, 6h, 24h}**:
  - Create panel: (segment_id, window_start) → future_crash_count
  - Shift labels by steps_ahead = H / W rows
  - Calculate sparsity metrics:
    - Zero rate: % of (segment, window) pairs with 0 crashes
    - Mean crashes per non-zero window
    - Variance-to-mean ratio (overdispersion indicator)
4. **Select H** based on:
  - If 1h zero rate > 90% → move to 6h
  - If 6h zero rate > 85% → move to 24h
  - If still sparse → consider ZIP (Zero-Inflated Poisson) or ZINB

### Critical Bug Prevention: steps_ahead Calculation

**IMPORTANT**: If panel window size is W hours and horizon is H hours:

- **steps_ahead = H / W** (number of rows to shift), NOT H raw hours
- **Validation check**: If H is not divisible by W, raise error or adjust H to nearest multiple

**Example**:

- Window size W = 1 hour
- Horizon H = 6 hours
- steps_ahead = 6 / 1 = 6 rows (shift by 6 windows)
- NOT: shift by 6 hours directly (would be wrong if windows are hourly)

**Pseudocode**:

```python
def calculate_steps_ahead(window_size_hours, horizon_hours):
    if horizon_hours % window_size_hours != 0:
        raise ValueError(f"Horizon {horizon_hours}h must be divisible by window size {window_size_hours}h")
    return horizon_hours // window_size_hours
```

### Handling Routes Longer Than H

**Strategy**:

1. Predict λ for first H hours using model
2. For remaining route duration, use historical average λ (baseline)
3. Optional: Dynamic updates during trip (re-predict every H hours)

**Implementation**:

- Route segments: [seg1, seg2, ..., segN]
- Travel times: [t1, t2, ..., tN]
- Cumulative time: cumsum([t1, t2, ..., tN])
- For segments where cumsum ≤ H: use predicted λ
- For segments where cumsum > H: use historical λ_avg

---

## Section C: Build Event-Level Crash Assignments

### Changes to `src/data_processing/spatial_join_fast.py`

**New Function**: `perform_spatial_join_event_level()`

**Requirements**:

1. **Stable segment_id**: Use CENTRELINE_ID whenever available
  ```python
   if 'CENTRELINE_ID' in road_segments.columns:
       road_segments['segment_id'] = road_segments['CENTRELINE_ID']
   else:
       road_segments['segment_id'] = road_segments.index.astype(str)
  ```
2. **Event-level output schema**:
  - `segment_id` (CENTRELINE_ID)
  - `event_datetime` (combine DATE + HOUR correctly)
  - `crash_type` ('collision' or 'ksi')
  - `is_ksi` (boolean)
  - `fatalities` (integer)
  - `geometry` (Point geometry of crash location)
3. **DATE + HOUR combination**:
  ```python
   # For collision data: OCC_DATE + OCC_HOUR
   event_datetime = pd.to_datetime(df['OCC_DATE']) + pd.to_timedelta(df['OCC_HOUR'], unit='h')

   # For KSI data: DATE + TIME (minutes since midnight)
   event_datetime = pd.to_datetime(df['DATE']) + pd.to_timedelta(df['TIME'], unit='m')
  ```
4. **Performance**: Avoid iterrows(), use vectorized operations
  ```python
   # Vectorized assignment using BallTree results
   crash_assignments = pd.DataFrame({
       'segment_id': segment_ids,  # from BallTree query
       'event_datetime': crash_points['event_datetime'],
       'crash_type': crash_points['crash_type'],
       'is_ksi': crash_points['is_ksi'],
       'fatalities': crash_points['fatalities'],
       'geometry': crash_points.geometry
   })
  ```
5. **Backward compatibility**: Keep existing `perform_spatial_join_fast()` for aggregate outputs

**Pseudocode**:

```python
def perform_spatial_join_event_level(collision_data, ksi_data, road_network):
    # 1. Ensure stable segment_id
    if 'CENTRELINE_ID' in road_network.columns:
        road_network['segment_id'] = road_network['CENTRELINE_ID']
    else:
        road_network['segment_id'] = road_network.index.astype(str)
    
    # 2. Combine DATE + HOUR for collision data
    collision_data['event_datetime'] = (
        pd.to_datetime(collision_data['OCC_DATE']) + 
        pd.to_timedelta(collision_data['OCC_HOUR'], unit='h')
    )
    collision_data['crash_type'] = 'collision'
    collision_data['is_ksi'] = False
    
    # 3. Combine DATE + TIME for KSI data
    ksi_data['event_datetime'] = (
        pd.to_datetime(ksi_data['DATE']) + 
        pd.to_timedelta(ksi_data['TIME'], unit='m')
    )
    ksi_data['crash_type'] = 'ksi'
    ksi_data['is_ksi'] = True
    
    # 4. Spatial join using BallTree (vectorized)
    road_proj = road_network.to_crs('EPSG:32617')
    road_coords = np.array([[p.x, p.y] for p in road_proj.geometry.centroid])
    tree = BallTree(road_coords, metric='euclidean')
    
    # Process collision points
    collision_coords = np.array([[p.x, p.y] for p in collision_data.to_crs('EPSG:32617').geometry])
    distances, indices = tree.query(collision_coords, k=1)
    within_buffer = distances.flatten() <= SPATIAL_BUFFER_DISTANCE
    
    # 5. Create event-level DataFrame (vectorized)
    collision_assignments = pd.DataFrame({
        'segment_id': road_network.iloc[indices.flatten()[within_buffer]]['segment_id'].values,
        'event_datetime': collision_data.loc[within_buffer, 'event_datetime'].values,
        'crash_type': 'collision',
        'is_ksi': False,
        'fatalities': collision_data.loc[within_buffer, 'FATALITIES'].values,
        'geometry': collision_data.loc[within_buffer].geometry.values
    })
    
    # Repeat for KSI data...
    
    # 6. Combine and return
    return pd.concat([collision_assignments, ksi_assignments], ignore_index=True)
```

---

## Section D: Build Temporal Panel Dataset (No Leakage)

### New Module: `src/feature_engineering/panel_builder.py`

**Purpose**: Build panel dataset indexed by (segment_id, window_start) with proper temporal ordering and no leakage

### Panel Structure

**Index**: (segment_id, window_start)

- `segment_id`: CENTRELINE_ID (stable identifier)
- `window_start`: datetime of window start (e.g., hourly: 2020-01-01 00:00:00)

**Features**:

1. **Static road features** (constant across time):
  - `segment_length` (meters)
  - `road_class_*` (one-hot encoded)
  - `FROM_INTERSECTION_ID`, `TO_INTERSECTION_ID` (for routing graph)
2. **Temporal indicators from window_start**:
  - `hour_of_day` (0-23)
  - `day_of_week` (0=Monday, 6=Sunday)
  - `is_weekend` (boolean)
  - `month` (1-12)
  - `season` (winter/spring/summer/fall)
  - `is_holiday` (optional, requires holiday calendar)
3. **Weather features** (from join):
  - `temperature` (Celsius)
  - `precipitation` (mm)
  - `visibility` (km)
  - `wind_speed` (km/h)
  - `weather_condition` (categorical: clear/rain/snow/etc.)
  - `is_missing_weather` (flag for missing data)
4. **Past-only lag features** (NO LEAKAGE):
  - `past_crash_count_1h` = crash_count.shift(1)  # Previous window
  - `past_crash_count_24h` = crash_count.shift(24)  # Same hour yesterday
  - `past_crash_count_7d` = crash_count.shift(168)  # Same hour last week
  - `rolling_mean_7d` = past_crash_count.rolling(168, min_periods=1).mean()  # On shifted data
  - `rolling_max_30d` = past_crash_count.rolling(720, min_periods=1).max()
5. **Future labels** (correctly shifted):
  - `future_crash_count` = crash_count.shift(-steps_ahead)  # Future window
  - `steps_ahead` = H / W (validated to be integer)

### Leakage Prevention Rules

**CRITICAL**: All lag/rolling features must be computed on PAST data only:

```python
# CORRECT: Shift first, then compute rolling
past_count = crash_count.shift(1)  # Move to past
rolling_mean = past_count.rolling(7).mean()  # Compute on past data

# WRONG: Compute rolling on current data
rolling_mean = crash_count.rolling(7).mean()  # Includes current window!
```

**Features to DELETE/DISABLE** (leak labels or simulated):

1. **Direct label leakage**:
  - `num_total_crashes` (current window - this is the label!)
  - `num_ksi_crashes` (current window)
  - `fatality_count` (current window)
  - `has_crashes`, `has_ksi`, `has_fatalities` (derived from current counts)
  - `crash_density`, `ksi_density` (uses current counts)
  - `severity_index`, `risk_score_raw` (uses current counts)
2. **Simulated features** (from `feature_creator.py`):
  - `time_of_day_morning`, `time_of_day_afternoon`, etc. (simulated, not real)
  - `weekend_crash_ratio` (simulated)
  - `season_*` counts (simulated)
  - `avg_hour` (simulated)
3. **Keep but fix**:
  - Road characteristics (static, OK)
  - Temporal indicators from window_start (OK, these are features not labels)

### Weather Join Strategy

**Join Keys**:

- Time: `window_start` rounded to nearest hour → `datetime_hour`
- Location: Segment centroid → `(lat, lon)` → grid cell `(lat_grid, lon_grid)`

**Caching**:

- Cache weather by `(datetime_hour, lat_grid, lon_grid)`
- Store in `outputs/cache/weather_cache.parquet`

**Missingness Handling**:

1. Forward fill within same station/cell (temporal continuity)
2. Spatial interpolation to nearest cell if available
3. Flag with `is_missing_weather = True` if no data available
4. Use median weather for segment if completely missing

**Pseudocode**:

```python
def build_panel_dataset(event_level_crashes, road_network, weather_data, 
                        window_size_hours=1, horizon_hours=6):
    # 1. Create time windows
    min_date = event_level_crashes['event_datetime'].min()
    max_date = event_level_crashes['event_datetime'].max()
    window_starts = pd.date_range(min_date, max_date, freq=f'{window_size_hours}H')
    
    # 2. Aggregate crashes to (segment_id, window_start)
    panel = event_level_crashes.groupby([
        'segment_id',
        pd.Grouper(key='event_datetime', freq=f'{window_size_hours}H', label='left')
    ]).agg({
        'is_ksi': 'sum',  # Count KSI crashes
        'fatalities': 'sum'
    }).reset_index()
    panel.rename(columns={'event_datetime': 'window_start'}, inplace=True)
    panel['crash_count'] = panel.groupby(['segment_id', 'window_start']).size()
    
    # 3. Create full panel (all segments × all windows)
    all_segments = road_network['segment_id'].unique()
    full_panel = pd.MultiIndex.from_product(
        [all_segments, window_starts],
        names=['segment_id', 'window_start']
    ).to_frame(index=False)
    panel = full_panel.merge(panel, on=['segment_id', 'window_start'], how='left')
    panel['crash_count'] = panel['crash_count'].fillna(0)
    
    # 4. Add static road features
    panel = panel.merge(
        road_network[['segment_id', 'segment_length', 'ROAD_CLASS', 
                     'FROM_INTERSECTION_ID', 'TO_INTERSECTION_ID']],
        on='segment_id',
        how='left'
    )
    
    # 5. Add temporal indicators from window_start
    panel['hour_of_day'] = panel['window_start'].dt.hour
    panel['day_of_week'] = panel['window_start'].dt.dayofweek
    panel['is_weekend'] = panel['day_of_week'].isin([5, 6])
    panel['month'] = panel['window_start'].dt.month
    panel['season'] = panel['month'].map(SEASON_MAPPING)
    
    # 6. Add weather features (join on datetime_hour + grid cell)
    panel['datetime_hour'] = panel['window_start'].dt.floor('H')
    panel['lat_grid'] = (panel['segment_centroid_lat'] // 0.01) * 0.01  # ~1km grid
    panel['lon_grid'] = (panel['segment_centroid_lon'] // 0.01) * 0.01
    panel = panel.merge(weather_data, on=['datetime_hour', 'lat_grid', 'lon_grid'], how='left')
    panel['is_missing_weather'] = panel['temperature'].isna()
    panel[['temperature', 'precipitation', 'visibility']] = (
        panel.groupby('segment_id')[['temperature', 'precipitation', 'visibility']]
        .ffill()  # Forward fill within segment
    )
    
    # 7. Add past-only lag features (CRITICAL: shift first!)
    panel = panel.sort_values(['segment_id', 'window_start'])
    panel['past_crash_count_1h'] = panel.groupby('segment_id')['crash_count'].shift(1)
    panel['past_crash_count_24h'] = panel.groupby('segment_id')['crash_count'].shift(24)
    panel['past_crash_count_7d'] = panel.groupby('segment_id')['crash_count'].shift(168)
    
    # Compute rolling on SHIFTED data (past-only)
    past_count = panel.groupby('segment_id')['crash_count'].shift(1)
    panel['rolling_mean_7d'] = past_count.groupby('segment_id').rolling(168, min_periods=1).mean().values
    panel['rolling_max_30d'] = past_count.groupby('segment_id').rolling(720, min_periods=1).max().values
    
    # 8. Add future labels (correctly shifted by steps_ahead)
    steps_ahead = horizon_hours // window_size_hours
    if horizon_hours % window_size_hours != 0:
        raise ValueError(f"Horizon {horizon_hours}h must be divisible by window {window_size_hours}h")
    
    panel['future_crash_count'] = panel.groupby('segment_id')['crash_count'].shift(-steps_ahead)
    
    # 9. Remove rows where future label is NaN (end of time series)
    panel = panel.dropna(subset=['future_crash_count'])
    
    return panel
```

---

## Section E: Temporal Evaluation (Matches Real World)

### Changes to `src/models/model_trainer.py`

**Replace random split with temporal split**:

```python
def temporal_train_test_split(panel_data, train_frac=0.6, val_frac=0.2, test_frac=0.2):
    # 1. Get unique ordered window_start values
    unique_windows = panel_data['window_start'].unique()
    unique_windows = np.sort(unique_windows)
    
    # 2. Allocate windows to splits
    n_windows = len(unique_windows)
    train_end = int(n_windows * train_frac)
    val_end = train_end + int(n_windows * val_frac)
    
    train_windows = unique_windows[:train_end]
    val_windows = unique_windows[train_end:val_end]
    test_windows = unique_windows[val_end:]
    
    # 3. Filter rows by window membership
    train_mask = panel_data['window_start'].isin(train_windows)
    val_mask = panel_data['window_start'].isin(val_windows)
    test_mask = panel_data['window_start'].isin(test_windows)
    
    train_data = panel_data[train_mask].copy()
    val_data = panel_data[val_mask].copy()
    test_data = panel_data[test_mask].copy()
    
    return train_data, val_data, test_data
```

### Rolling Window Evaluation / Time-Series CV

**Implementation**:

```python
def rolling_window_evaluation(model, panel_data, n_splits=5):
    unique_windows = np.sort(panel_data['window_start'].unique())
    window_size = len(unique_windows) // (n_splits + 1)
    
    results = []
    for i in range(n_splits):
        # Train on windows [0, ..., (i+1)*window_size)
        train_end = (i + 1) * window_size
        train_windows = unique_windows[:train_end]
        
        # Test on next window
        test_start = train_end
        test_end = test_start + window_size
        test_windows = unique_windows[test_start:test_end]
        
        train_data = panel_data[panel_data['window_start'].isin(train_windows)]
        test_data = panel_data[panel_data['window_start'].isin(test_windows)]
        
        # Train and evaluate
        model.fit(train_data[X_cols], train_data['future_crash_count'])
        predictions = model.predict(test_data[X_cols])
        
        metrics = calculate_metrics(test_data['future_crash_count'], predictions)
        metrics['window_start'] = test_windows[0]
        metrics['window_end'] = test_windows[-1]
        results.append(metrics)
    
    return pd.DataFrame(results)
```

**Reporting**:

- Metrics per window: MAE, RMSE, Poisson deviance, zero-inflation rate
- Aggregated: mean ± std across windows
- Validation: "Predicting future crashes" = test windows are strictly after train windows

---

## Section F: Modeling Approach

### Model Family: Negative Binomial Regression

**Rationale**:

- Target: crash count per hour (count data)
- Overdispersion: Crash counts typically have variance > mean
- Negative Binomial handles overdispersion via dispersion parameter α
- Alternative: Poisson if variance ≈ mean (test with dispersion test)

**Implementation**:

```python
from sklearn.ensemble import GradientBoostingRegressor
from scipy.stats import nbinom

# Option 1: GBDT with Poisson/Negative Binomial objective
model = GradientBoostingRegressor(
    loss='poisson',  # or custom negative binomial loss
    n_estimators=200,
    max_depth=5
)

# Option 2: GLM with Negative Binomial (statsmodels)
from statsmodels.discrete.discrete_model import NegativeBinomial
model = NegativeBinomial(endog=y, exog=X, loglike_method='nb2')
```

**Output Conversion**:

- Model predicts: λ (crashes/hour)
- For routing: 
  - Edge expected crashes = λ · t (where t = travel_time_hours)
  - Route expected crashes = Σ(λᵢ · tᵢ)
  - Route probability = 1 - exp(-Σ(λᵢ · tᵢ))

**SMOTE is NOT appropriate**: SMOTE is for classification, not count regression. Use:

- Class weights (if treating as classification)
- Or better: Use count regression directly (Poisson/NegBin)

**Sparsity Handling**:

- If >90% zeros: Consider Zero-Inflated Negative Binomial (ZINB)
- Or: Use hurdle model (binary classifier + count regressor)

---

## Section G: Calibration and Confidence

### Calibration for Count Models

**If model outputs λ (rate)**:

1. Convert to traversal probability: P = 1 - exp(-λ · t)
2. Bin predicted P vs observed frequency
3. Apply isotonic calibration if needed: `calibrated_P = calibrator.transform(P)`

**Implementation**:

```python
from sklearn.isotonic import IsotonicRegression

# 1. Get predictions on validation set
lambda_pred = model.predict(X_val)
P_pred = 1 - np.exp(-lambda_pred * travel_times_val)

# 2. Get observed outcomes (binary: crash or no crash)
y_observed = (y_val > 0).astype(int)

# 3. Calibrate
calibrator = IsotonicRegression(out_of_bounds='clip')
calibrator.fit(P_pred, y_observed)
P_calibrated = calibrator.transform(P_pred)

# 4. Save calibrator with model
model_data = {
    'model': model,
    'calibrator': calibrator,
    'feature_columns': feature_columns
}
```

**Inference**:

```python
def predict_crash_likelihood(model, calibrator, features, travel_time_hours):
    lambda_pred = model.predict(features)
    P_raw = 1 - np.exp(-lambda_pred * travel_time_hours)
    P_calibrated = calibrator.transform(P_raw)
    return P_calibrated, lambda_pred
```

**DO NOT** assume classifier `predict_proba()` - model outputs λ, not probabilities directly.

---

## Section H: Routing Integration Requirements

### Graph Construction

**New Module**: `src/routing/road_graph.py`

**Graph Structure**:

- **Nodes**: Intersections (FROM_INTERSECTION_ID, TO_INTERSECTION_ID)
- **Edges**: Road segments with:
  - `segment_id` = CENTRELINE_ID
  - `geometry` (LineString)
  - `length` (meters)
  - `road_class`
  - `oneway` (from ONEWAY_DIR_CODE)

**Implementation**:

```python
import networkx as nx

def build_road_graph(road_network):
    G = nx.DiGraph()  # Directed graph (one-way streets)
    
    for _, segment in road_network.iterrows():
        from_node = segment['FROM_INTERSECTION_ID']
        to_node = segment['TO_INTERSECTION_ID']
        segment_id = segment['CENTRELINE_ID']
        
        # Add edge
        G.add_edge(
            from_node,
            to_node,
            segment_id=segment_id,
            geometry=segment.geometry,
            length=segment['segment_length'],
            road_class=segment['ROAD_CLASS']
        )
        
        # Handle one-way: if two-way, add reverse edge
        if segment['ONEWAY_DIR_CODE'] != 'ONE_WAY':
            G.add_edge(
                to_node,
                from_node,
                segment_id=segment_id,  # Same segment, reverse direction
                geometry=segment.geometry.reverse(),
                length=segment['segment_length'],
                road_class=segment['ROAD_CLASS']
            )
    
    return G
```

### Travel Time Estimation

**Strategy**: Default speeds by road class (if no speed limits)

```python
DEFAULT_SPEEDS_KMH = {
    'arterial': 50,
    'collector': 40,
    'local': 30,
    'minor_arterial': 45
}

def estimate_travel_time(segment_length_m, road_class):
    speed_kmh = DEFAULT_SPEEDS_KMH.get(road_class, 40)
    speed_ms = speed_kmh / 3.6
    travel_time_seconds = segment_length_m / speed_ms
    return travel_time_seconds / 3600  # Convert to hours
```

### Risk-to-Cost Conversion

**Edge Weight Formula**:

```
edge_cost_hours = travel_time_hours + beta_hours_per_expected_crash * expected_crashes
```

Where:

- `travel_time_hours` = segment_length / speed (hours)
- `expected_crashes` = λ · travel_time_hours (from model)
- `beta` = user preference parameter (tunable, e.g., 0.1 = 0.1 hours per expected crash)

**Units**: All in hours (time-equivalent cost)

**Route Aggregation**:

```python
def calculate_route_risk(route_segments, lambda_predictions, travel_times):
    route_expected_crashes = sum(λ * t for λ, t in zip(lambda_predictions, travel_times))
    route_probability = 1 - np.exp(-route_expected_crashes)
    return route_expected_crashes, route_probability
```

### Routing Algorithm

**Fastest Route**: Dijkstra on `travel_time_hours`

**Safer Route**: Dijkstra on `edge_cost_hours` (with beta > 0)

**Implementation**:

```python
def find_fastest_route(graph, start_node, end_node, lambda_predictions, beta=0.1):
    # Edge weights: travel_time only
    edge_weights = {
        (u, v): data['travel_time_hours']
        for u, v, data in graph.edges(data=True)
    }
    path = nx.dijkstra_path(graph, start_node, end_node, weight=edge_weights)
    return path

def find_safer_route(graph, start_node, end_node, lambda_predictions, travel_times, beta=0.1):
    # Edge weights: travel_time + beta * expected_crashes
    edge_weights = {}
    for u, v, data in graph.edges(data=True):
        segment_id = data['segment_id']
        travel_time = data['travel_time_hours']
        lambda_pred = lambda_predictions.get(segment_id, 0)
        expected_crashes = lambda_pred * travel_time
        edge_weights[(u, v)] = travel_time + beta * expected_crashes
    
    path = nx.dijkstra_path(graph, start_node, end_node, weight=edge_weights)
    return path
```

### UI Outputs

**Required Information**:

1. **Avoided segments**: Segments in fastest route but not in safer route
2. **Risk drivers**: Top features contributing to high λ (SHAP values)
3. **Time/risk tradeoff**:
  - Fastest route: time T_fast, risk R_fast
  - Safer route: time T_safe, risk R_safe
  - Tradeoff: ΔT = T_safe - T_fast, ΔR = R_fast - R_safe

### Snap-to-Graph

**Node Geometry Construction**:

**Option 1**: Derive from segment endpoints

```python
def build_node_geometry(graph, road_network):
    node_coords = {}
    for segment_id, segment in road_network.iterrows():
        from_node = segment['FROM_INTERSECTION_ID']
        to_node = segment['TO_INTERSECTION_ID']
        
        # Get segment endpoints
        geom = segment.geometry
        start_point = Point(geom.coords[0])
        end_point = Point(geom.coords[-1])
        
        # Use first segment's endpoint for each node
        if from_node not in node_coords:
            node_coords[from_node] = start_point
        if to_node not in node_coords:
            node_coords[to_node] = end_point
    
    return node_coords
```

**Option 2**: Use intersections dataset if available

**Snap Logic**:

```python
def snap_to_graph(user_point, graph, node_coords, max_distance_m=100):
    # Find nearest node within max_distance
    min_dist = float('inf')
    nearest_node = None
    
    for node_id, node_point in node_coords.items():
        dist = user_point.distance(node_point) * 111000  # Approx meters (WGS84)
        if dist < max_distance_m and dist < min_dist:
            min_dist = dist
            nearest_node = node_id
    
    if nearest_node is None:
        raise ValueError(f"No node within {max_distance_m}m")
    
    return nearest_node
```

### Maps Export

**Waypoint Constraints**: Google/Apple Maps ~25 waypoints max

**Simplification Strategy**:

1. **Douglas-Peucker**: Simplify route geometry to reduce points
2. **Cap points**: If still >25, sample evenly spaced waypoints
3. **Segment grouping**: Group consecutive segments into waypoint regions

**Implementation**:

```python
from shapely.ops import simplify

def simplify_route_for_export(route_geometry, max_waypoints=25):
    # 1. Simplify geometry
    simplified = simplify(route_geometry, tolerance=0.0001)  # ~10m tolerance
    
    # 2. Extract coordinates
    coords = list(simplified.coords)
    
    # 3. If still too many, sample evenly
    if len(coords) > max_waypoints:
        indices = np.linspace(0, len(coords)-1, max_waypoints, dtype=int)
        coords = [coords[i] for i in indices]
    
    return coords
```

**External App Re-routing Mitigation**:

- Note: External apps may re-route
- Mitigation: Export as waypoints, not full route
- User education: "Route may vary in navigation app"

---

## Section I: CRS / Distance Verification

### CRS Checks

**Add to `src/data_processing/spatial_join_fast.py**`:

```python
def verify_crs_and_distance(road_network, buffer_distance_m=20):
    # 1. Check CRS is set
    if road_network.crs is None:
        raise ValueError("Road network CRS is not set")
    
    # 2. Convert to projected CRS (meters)
    road_proj = road_network.to_crs('EPSG:32617')  # UTM Zone 17N
    
    # 3. Verify buffer distance is in meters
    test_point = road_proj.geometry.iloc[0].centroid
    test_buffer = test_point.buffer(buffer_distance_m)
    buffer_area = test_buffer.area
    
    # Expected area for 20m buffer: π * 20² ≈ 1256 m²
    expected_area = np.pi * buffer_distance_m ** 2
    if abs(buffer_area - expected_area) > 100:  # 100 m² tolerance
        raise ValueError(f"Buffer distance appears incorrect. Expected area ~{expected_area:.0f} m², got {buffer_area:.0f} m²")
    
    logger.info(f"CRS verified: {road_network.crs}, buffer distance {buffer_distance_m}m confirmed")
    return True
```

### Test Case

```python
def test_20m_distance_check():
    # Create test data
    point1 = Point(0, 0)
    point2 = Point(0.00018, 0)  # ~20m at Toronto latitude
    gdf = gpd.GeoDataFrame({'geometry': [point1, point2]}, crs='EPSG:4326')
    gdf_proj = gdf.to_crs('EPSG:32617')
    
    # Calculate distance
    dist = gdf_proj.geometry.iloc[0].distance(gdf_proj.geometry.iloc[1])
    
    # Verify ~20m
    assert 19.5 < dist < 20.5, f"Distance should be ~20m, got {dist:.2f}m"
    print("✓ 20m distance check passed")
```

---

## Section J: Acceptance Criteria

### "Done When" Criteria

1. **No Leakage**:
  - Test: No features use current window crash counts
  - Test: All lag features computed on shifted data
  - Test: Future labels shifted by correct steps_ahead
2. **Temporal Test Performance**:
  - Test windows are strictly after train windows
  - Report metrics per window (MAE, RMSE, Poisson deviance)
  - Report aggregated mean ± std
3. **Inference on Unseen Future**:
  - Inference runs on windows not in training set
  - No scoring of training rows during inference
  - Validation: Check window_start of inference data > max(train window_start)
4. **Interpretable Outputs**:
  - Model outputs λ (crashes/hour) under current conditions
  - Conversion to traversal probability: P = 1 - exp(-λ · t)
  - Route aggregation: Σ(λᵢ · tᵢ) for expected crashes
5. **Routing Edge Weights**:
  - Edge weights computed from λ + travel_time with correct units
  - Route aggregation uses expected crashes formula
  - Beta parameter tunable
6. **Stable Segment IDs**:
  - CENTRELINE_ID used end-to-end (data processing → model → routing)
  - No segment_id instability (test: same CENTRELINE_ID → same predictions)
7. **Snap-to-Graph**:
  - Node geometry constructed (from endpoints or intersections dataset)
  - Max snap distance enforced (default 100m)
  - Test: User point within 100m snaps correctly
8. **Maps Export**:
  - Waypoint count ≤ 25
  - Douglas-Peucker simplification applied
  - Route geometry simplified appropriately

---

## Section K: Deliverables

### Files to Modify

1. `**src/data_processing/spatial_join_fast.py**`:
  - Add `perform_spatial_join_event_level()` function
  - Use CENTRELINE_ID as stable segment_id
  - Combine DATE + HOUR correctly
  - Keep existing `perform_spatial_join_fast()` for backward compatibility
  - Add CRS verification function
2. `**src/feature_engineering/panel_builder.py**` (NEW):
  - `build_panel_dataset()` - main function
  - Vectorized operations (groupby/shift/rolling)
  - Past-only lag features (shift first!)
  - Future labels with correct steps_ahead
  - Weather join logic
  - Temporal split function
3. `**src/feature_engineering/feature_creator.py**`:
  - DELETE simulated temporal features
  - Keep only static road features
  - Add temporal indicators from window_start (not simulated)
4. `**src/feature_engineering/label_generator.py**`:
  - DEPRECATE (rule-based labeling not used for prediction)
  - Keep for backward compatibility/analysis only
5. `**src/models/model_trainer.py**`:
  - Replace `train_test_split` with `temporal_train_test_split`
  - Change target from `risk_label` to `future_crash_count` (or `crash_rate`)
  - Change model from RandomForestClassifier to NegativeBinomial/GBDT with count objective
  - Remove SMOTE (not appropriate for count regression)
  - Add calibration support
  - Add rolling window evaluation
6. `**src/models/model_evaluator.py**`:
  - Update metrics for count regression (MAE, RMSE, Poisson deviance)
  - Add per-window evaluation
  - Add calibration plots
7. `**src/routing/road_graph.py**` (NEW):
  - `build_road_graph()` - construct NetworkX graph
  - `estimate_travel_time()` - default speeds by road class
  - `find_fastest_route()` - Dijkstra on travel_time
  - `find_safer_route()` - Dijkstra on combined cost
  - `calculate_route_risk()` - aggregate route expected crashes
  - `snap_to_graph()` - snap user point to nearest node
  - `build_node_geometry()` - derive node coordinates
8. `**src/routing/maps_export.py**` (NEW):
  - `simplify_route_for_export()` - Douglas-Peucker + waypoint cap
  - `export_to_google_maps()` - format waypoints for Google Maps
  - `export_to_apple_maps()` - format waypoints for Apple Maps
9. `**backend-api/app.py**`:
  - Add routing endpoints: `/api/route/fastest`, `/api/route/safer`
  - Add risk prediction endpoint with time/weather: `/api/risk-prediction/live`
  - Update inference to use panel features (not leaked features)
  - Add calibration application in predictions
10. `**config.py**`:
  - Add routing parameters: `DEFAULT_SPEEDS_KMH`, `BETA_RISK_WEIGHT`
    - Add panel parameters: `WINDOW_SIZE_HOURS`, `HORIZON_HOURS`
    - Add weather parameters: `WEATHER_CACHE_PATH`, `WEATHER_GRID_SIZE`
11. `**run_risk_analysis.py**`:
  - Update pipeline to use event-level joins
    - Update pipeline to build panel dataset
    - Update pipeline to use temporal splits
    - Update pipeline to train count regression model

### Pseudocode Summary

**Event-level join output (DATE+HOUR combined)**:

```python
# In spatial_join_fast.py
collision_data['event_datetime'] = (
    pd.to_datetime(collision_data['OCC_DATE']) + 
    pd.to_timedelta(collision_data['OCC_HOUR'], unit='h')
)
```

**Panel builder (vectorized + correct steps_ahead + past-only rolling)**:

```python
# In panel_builder.py
panel = event_crashes.groupby(['segment_id', pd.Grouper(key='event_datetime', freq='1H')]).size()
panel = panel.sort_values(['segment_id', 'window_start'])
past_count = panel.groupby('segment_id')['crash_count'].shift(1)  # Shift first!
rolling_mean = past_count.groupby('segment_id').rolling(168).mean()  # Then roll
steps_ahead = horizon_hours // window_size_hours
panel['future_crash_count'] = panel.groupby('segment_id')['crash_count'].shift(-steps_ahead)
```

**Temporal split logic (by unique ordered windows)**:

```python
# In model_trainer.py
unique_windows = np.sort(panel_data['window_start'].unique())
train_end = int(len(unique_windows) * 0.6)
train_windows = unique_windows[:train_end]
test_windows = unique_windows[train_end:]
train_data = panel_data[panel_data['window_start'].isin(train_windows)]
```

**Inference prediction (weather/time conditioned + calibration)**:

```python
# In backend-api/app.py
lambda_pred = model.predict(features_with_current_time_weather)
P_raw = 1 - np.exp(-lambda_pred * travel_time_hours)
P_calibrated = calibrator.transform(P_raw)
```

**Risk-to-cost conversion (units explicit)**:

```python
# In road_graph.py
travel_time_hours = segment_length_m / (speed_kmh / 3.6) / 3600
expected_crashes = lambda_pred * travel_time_hours  # crashes
edge_cost_hours = travel_time_hours + beta * expected_crashes  # hours (time-equivalent)
```

**Snap-to-graph (including node coordinates)**:

```python
# In road_graph.py
node_coords = {node_id: segment_endpoint for segment in road_network}
nearest_node = min(node_coords.items(), 
                   key=lambda x: user_point.distance(x[1]) * 111000)
if distance > max_distance_m: raise ValueError("Too far from road network")
```

**Maps export (waypoint cap + simplification)**:

```python
# In maps_export.py
simplified = simplify(route_geometry, tolerance=0.0001)
coords = list(simplified.coords)
if len(coords) > 25:
    coords = [coords[i] for i in np.linspace(0, len(coords)-1, 25, dtype=int)]
```

---

## Implementation Phases

### Phase 1: Data Foundation (Weeks 1-2)

- Event-level crash assignments
- Panel dataset construction
- Weather integration
- CRS verification

### Phase 2: Model Refactor (Weeks 3-4)

- Temporal splits
- Count regression model
- Calibration
- Evaluation metrics

### Phase 3: Routing Infrastructure (Weeks 5-6)

- Graph construction
- Routing algorithms
- Snap-to-graph
- Maps export

### Phase 4: Integration & Testing (Weeks 7-8)

- Backend API updates
- End-to-end testing
- Acceptance criteria validation
- Documentation

---

## Risks and Mitigations

1. **Risk**: Data sparsity makes 1h horizon unrealistic
  - **Mitigation**: Implement sparsity analysis, fallback to 6h/24h or ZIP/ZINB
2. **Risk**: Weather data unavailable or incomplete
  - **Mitigation**: Missingness flags, forward fill, median imputation
3. **Risk**: Routing graph construction fails (missing intersection IDs)
  - **Mitigation**: Derive nodes from segment endpoints, validate graph connectivity
4. **Risk**: Performance degradation with panel dataset size
  - **Mitigation**: Chunked processing, efficient indexing, consider sampling
5. **Risk**: Model calibration fails (poor fit)
  - **Mitigation**: Validate calibration on held-out validation set, fallback to uncalibrated

---

## Backward Compatibility

**Kept for compatibility**:

- `perform_spatial_join_fast()` - aggregate outputs still available
- `generate_risk_labels()` - for analysis/reporting, not prediction
- Existing model files - can load old models for comparison

**Breaking changes**:

- New model outputs λ instead of risk_label
- New features required (weather, temporal indicators)
- New data format (panel dataset)

**Migration path**:

- Old models marked as deprecated
- New models trained alongside old for comparison
- Gradual transition in API endpoints

