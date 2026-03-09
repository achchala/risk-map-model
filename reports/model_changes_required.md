# Predictive Model — Required Changes

## Context

The temporal crash prediction model on the `feature/weather-hourly-model` branch was not functioning as a predictive model. An audit of the trained model revealed it was memorizing individual road segments via raw intersection IDs and coordinates rather than learning generalizable patterns from road characteristics, weather, and temporal signals. This document details the changes required to make the model genuinely predictive.

---

## Problem Summary

The model used **17 features**, of which the majority were broken or counterproductive:

| Issue | Features affected | Impact |
|-------|------------------|--------|
| **Segment identity memorization** | `FROM_INTERSECTION_ID`, `TO_INTERSECTION_ID`, `segment_centroid_lat`, `segment_centroid_lon` | Model looked up "which segment is this" instead of learning generalizable risk patterns |
| **String columns silently zeroed** | `ROAD_CLASS`, `season` | Converted to 0.0 for every row by `pd.to_numeric(errors="coerce")` — carried no information |
| **One-hot road class not reaching model** | `road_class_*` columns | Built in the panel but not present in the saved model's feature list |
| **No temporal variation** | `hour_of_day`, `day_of_week`, `is_weekend` | Test set had a single unique value for each (all hour 13, all Saturday) due to stale weekly-window training |
| **No weather data** | `is_missing_weather` | 100% True — `historicalweather.csv` did not exist |
| **No traffic volume data** | None present | `model_dataset.csv` did not exist |
| **Lag features ~99% zeros** | `past_crash_count_1h/24h/7d` | At hourly granularity, crashes are too rare per segment for raw lags to carry signal |

The model was effectively a lookup table keyed by segment ID, not a predictive model.

---

## Required Changes

### 1. Fix Feature Selection (`src/models/model_trainer.py`)

**File:** `src/models/model_trainer.py` — `TemporalCountModelTrainer.prepare_panel_features()`

Add the following to the `exclude` set to prevent segment memorization:

- `FROM_INTERSECTION_ID`, `TO_INTERSECTION_ID` — raw identifiers that allow the model to memorize specific segments
- `segment_centroid_lat`, `segment_centroid_lon` — spatial coordinates used only for weather grid joins, not as predictive features
- `ROAD_CLASS`, `season` — raw string columns replaced by proper encodings (one-hot and integer respectively)
- `hour_of_day`, `day_of_week`, `month` — raw integers replaced by cyclical sin/cos encodings

Add logging of the final feature column list for transparency and debugging.

---

### 2. Fix Temporal Feature Encoding (`src/feature_engineering/panel_builder.py`)

**File:** `src/feature_engineering/panel_builder.py` — `_add_temporal_indicators()`

Replace raw integer temporal features with cyclical sin/cos encodings so the model understands periodicity (hour 23 is close to hour 0):

- `hour_sin`, `hour_cos` — sin/cos of `hour_of_day / 24 * 2π`
- `dow_sin`, `dow_cos` — sin/cos of `day_of_week / 7 * 2π`
- `month_sin`, `month_cos` — sin/cos of `month / 12 * 2π`
- `season_int` — integer encoding (0–3) replacing the string `season` column

---

### 3. Add Historical Crash Profile Features (`src/feature_engineering/panel_builder.py`)

**File:** `src/feature_engineering/panel_builder.py` — new function `_compute_historical_crash_profiles()`

At hourly granularity, raw lag features (`past_crash_count_1h`) are ~99% zeros because crashes are rare per segment per hour. Add aggregated historical features that carry meaningful signal:

- `hist_crashes_per_year` — average annual crash count for the segment (baseline risk level)
- `hist_crash_hour_ratio` — fraction of the segment's historical crashes in the same 6-hour time bucket (captures time-of-day risk profile)
- `hist_crash_weekend_ratio` — fraction of crashes on weekends for the segment

Call this function from `build_weekly_sampled_future_panel()` and `build_latest_window_inference_panel()`.

---

### 4. Remove Identity Proxies from Static Features (`src/feature_engineering/panel_builder.py`)

**File:** `src/feature_engineering/panel_builder.py` — `build_weekly_sampled_future_panel()`, `build_latest_window_inference_panel()`, `build_panel_dataset()`

Remove `FROM_INTERSECTION_ID` and `TO_INTERSECTION_ID` from the static feature columns attached to the panel. These are raw identifiers that let the model memorize individual segments. Keep only genuinely informative road attributes:

- `segment_length`, `ROAD_CLASS` (for one-hot encoding), `is_oneway`
- `from_intersection_degree`, `to_intersection_degree` (graph connectivity — informative, not identifying)
- Traffic volume columns from `model_dataset.csv` (when available)

Use a shared constant `_TRAFFIC_VOLUME_COLS` for the traffic column list to avoid duplication across the three panel builder functions.

---

### 5. Fix Weather Data Loader (`src/data_processing/data_loader.py`)

**File:** `src/data_processing/data_loader.py` — `load_historical_weather()`

The existing loader has bugs that prevent it from reading the NCEI/NOAA CSV format:

- Remove `skiprows=1` — the file has the header on row 1 (no extra row to skip)
- Use `"DATE"` column (uppercase) instead of `"Date"`
- Handle `SNWD` (snow depth) in addition to `SNOW` (snowfall)
- Add `AWND` (average wind speed) parsing with mph → m/s conversion
- Add derived binary features: `is_freezing` (temperature ≤ 0°C), `is_precip` (precipitation > 0)
- Output columns: `datetime_hour`, `temperature`, `precipitation`, `snow_depth_mm`, `wind_speed`, `is_freezing`, `is_precip`

Update `_attach_weather_features()` in `panel_builder.py` to recognize the new column names (`snow_depth_mm`, `is_freezing`, `is_precip`).

---

### 6. Fix Traffic Volume Data Loader (`src/data_processing/data_loader.py`)

**File:** `src/data_processing/data_loader.py` — `load_model_dataset()`

Update to handle the actual `model_dataset.csv` format from Toronto Open Data:

- Explicitly exclude leakage columns: `crash_count`, `crash_rate`, `segment_length` (road network already has a geometry-derived version)
- Pick up additional useful columns: `avg_wkdy_am_peak_vol`, `avg_wkdy_pm_peak_vol`, `avg_95th_percentile_speed`, `avg_heavy_pct`, `log_volume`
- Use a whitelist approach (`SAFE_COLS`) rather than blacklist to prevent future leakage

---

### 7. Retrain the Model

**Command:** `python train_temporal_model.py`

All code changes are validated but the saved model artifacts are stale from the old broken pipeline. Retraining produces:

- `outputs/models/toronto_temporal_count_model.pkl` — updated model with ~39 features
- `outputs/reports/panel_latest.parquet` — inference panel snapshot for the API
- `outputs/reports/temporal_model_test_results.npz` — test set diagnostics
- `outputs/reports/temporal_model_test_set_with_pred.parquet` — test predictions for inspection

---

## Resulting Feature Set (~39 features)

| Category | Count | Features |
|----------|-------|----------|
| **Road type** | 5 | `road_class_Local`, `road_class_Collector`, `road_class_Minor_Arterial`, `road_class_Major_Arterial`, `road_class_Expressway` |
| **Road properties** | 4 | `segment_length`, `is_oneway`, `from_intersection_degree`, `to_intersection_degree` |
| **Traffic volume** | 9 | `avg_daily_vol`, `avg_speed`, `avg_85th_percentile_speed`, `avg_95th_percentile_speed`, `exposure`, `avg_wkdy_am_peak_vol`, `avg_wkdy_pm_peak_vol`, `avg_heavy_pct`, `log_volume` |
| **Temporal (cyclical)** | 7 | `hour_sin`, `hour_cos`, `dow_sin`, `dow_cos`, `month_sin`, `month_cos`, `season_int`, `is_weekend` |
| **Weather** | 6 | `temperature`, `precipitation`, `snow_depth_mm`, `wind_speed`, `is_freezing`, `is_precip` |
| **Historical crash profile** | 3 | `hist_crashes_per_year`, `hist_crash_hour_ratio`, `hist_crash_weekend_ratio` |
| **Lag features** | 5 | `past_crash_count_1h`, `past_crash_count_24h`, `past_crash_count_7d`, `rolling_mean_24h`, `rolling_max_24h` |

---

## Optional Future Improvement

**Upgrade to hourly weather data.** The current weather is daily NOAA data expanded to 24 identical hourly rows per day. Environment Canada publishes actual hourly observations for Toronto Pearson airport. Replacing the daily-expanded data with real hourly observations would give the model true hour-to-hour weather variation (e.g., "it's raining right now" vs "it rained this morning") — the single biggest remaining improvement for hourly prediction credibility.
