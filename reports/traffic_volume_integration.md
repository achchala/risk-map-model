# Traffic Volume Data Integration

## Data Source

**File:** `data/model_dataset.csv`
**Source:** City of Toronto Open Data — Traffic Speed/Volume studies
**Coverage:** 11,562 road segments out of 65,133 total (17.8%)
**Date added:** March 9, 2026

---

## Raw Data Overview

The CSV contains 16 columns per segment. After filtering for safety (excluding crash-derived columns that would leak the prediction target), 9 traffic features are retained:

| Column | Population | Description |
|--------|-----------|-------------|
| `avg_daily_vol` | 11,562 (100%) | Average daily traffic volume across all days. Range: 3–64,750 vehicles/day |
| `avg_speed` | 7,187 (62.2%) | Average recorded speed in km/h. Range: 10–118 km/h |
| `avg_85th_percentile_speed` | 7,187 (62.2%) | Speed below which 85% of vehicles travel. Range: 16–148 km/h |
| `avg_95th_percentile_speed` | 7,187 (62.2%) | Speed below which 95% of vehicles travel. Range: 21–163 km/h |
| `exposure` | 11,562 (100%) | Vehicle-km per year (volume × segment length × 365). Standard risk normalization metric |
| `avg_wkdy_am_peak_vol` | 11,541 (99.8%) | Average weekday AM peak hour volume (morning rush). Range: 0–4,870 |
| `avg_wkdy_pm_peak_vol` | 11,543 (99.8%) | Average weekday PM peak hour volume (evening rush). Range: 0–4,260 |
| `avg_heavy_pct` | 163 (1.4%) | Percentage of heavy vehicles (trucks). Very sparse — mostly 0 |
| `log_volume` | 11,562 (100%) | Natural log of `avg_daily_vol`. Reduces skew for modeling |

### Columns excluded from the model

| Column | Reason |
|--------|--------|
| `crash_count` | Direct target leakage — this is what the model predicts |
| `crash_rate` | Derived from crash_count — same leakage risk |
| `segment_length` | Already exists on the road network (computed from geometry, more accurate) |
| `log_exposure` | Derived from exposure — redundant |
| `avg_weekday_daily_vol` | Nearly identical to `avg_daily_vol` — adds no signal |
| `avg_weekend_daily_vol` | Only 7.5% populated — too sparse to be useful |

---

## Code Changes

### `src/data_processing/data_loader.py` — `load_model_dataset()`

- Switched from a blacklist to a **whitelist** (`SAFE_COLS`) approach for column selection. Only explicitly approved traffic/speed columns pass through.
- Added 5 new columns to the whitelist: `avg_wkdy_am_peak_vol`, `avg_wkdy_pm_peak_vol`, `avg_95th_percentile_speed`, `avg_heavy_pct`, `log_volume`.
- `segment_length` is excluded because the road network already has a geometry-derived version that is more accurate than the CSV approximation.

### `src/data_processing/data_loader.py` — `merge_model_dataset_into_road_network()`

No changes required. The existing left-join by `segment_id` handles the merge correctly. Segments without traffic data (53,571 of 65,133) receive 0 for all traffic columns — the model can distinguish "no data" (0) from measured low-volume roads (small positive values).

### `src/feature_engineering/panel_builder.py`

- Introduced a shared constant `_TRAFFIC_VOLUME_COLS` listing all 10 potential traffic columns. This replaces three separate hardcoded lists in `build_panel_dataset()`, `build_weekly_sampled_future_panel()`, and `build_latest_window_inference_panel()`.
- Each panel builder function iterates `_TRAFFIC_VOLUME_COLS` and includes any column that exists on the road network after the merge. This means the pipeline automatically adapts to whatever traffic columns are available without code changes.

### `src/models/model_trainer.py`

No changes required. The model trainer's `prepare_panel_features()` exclude set does not mention any traffic columns, so they pass through to the model automatically.

---

## How This Improves the Model

### Before: no traffic differentiation

The model had no way to distinguish between:
- A quiet residential street with 500 vehicles/day
- A major arterial carrying 40,000 vehicles/day
- A highway with high-speed traffic

All roads of the same `ROAD_CLASS` received identical predictions regardless of actual traffic exposure.

### After: traffic exposure as a core predictor

The model can now learn patterns like:
- **High volume + high speed = elevated risk.** A 6-lane arterial at 60 km/h with 30,000 daily vehicles has fundamentally different crash dynamics than a 2-lane local road at 30 km/h with 2,000 vehicles.
- **Peak-hour congestion signal.** `avg_wkdy_am_peak_vol` and `avg_wkdy_pm_peak_vol` indicate how congested the segment gets during rush hours. Combined with the cyclical hour-of-day encoding, the model can predict higher risk on high-peak-volume segments specifically during rush hours.
- **Speed variance as a risk factor.** The difference between `avg_speed` and `avg_85th/95th_percentile_speed` captures speed variance — a known crash predictor. Segments where most drivers go 40 km/h but 5% go 70 km/h are riskier than segments where everyone goes 50 km/h.
- **Exposure normalization.** The `exposure` column (vehicle-km/year) lets the model distinguish between a segment that's dangerous *per vehicle* versus one that simply has a lot of traffic. This is the standard metric in traffic safety research for risk-adjusted crash rates.

### Feature count progression

| Stage | Feature count | What changed |
|-------|--------------|--------------|
| Original broken model | 17 | Identity proxies, broken strings, no weather, no traffic |
| After feature pipeline fix | 25 | Proper road class, cyclical time, historical profiles |
| After weather integration | 30 | Temperature, precipitation, snow, freezing flag |
| **After traffic volume integration** | **~39** | Daily volume, speed, peak volumes, exposure |

---

## Validation

The integration was validated end-to-end:

1. **Loader test:** `load_model_dataset()` correctly reads 11,562 segments with 9 traffic columns, excluding all crash-derived columns.
2. **Merge test:** Left-join onto the 65,133-segment road network produces no column collisions (`segment_length` is not duplicated).
3. **Panel test:** Traffic columns flow through the panel builder and appear in the model's feature list.
4. **No leakage:** `crash_count` and `crash_rate` from the CSV are never included in any feature set.

---

## Remaining Step

The model must be retrained (`python train_temporal_model.py`) for these features to take effect. The saved model artifacts (`toronto_temporal_count_model.pkl`, `panel_latest.parquet`) are still from the old pipeline and do not include traffic volume features.
