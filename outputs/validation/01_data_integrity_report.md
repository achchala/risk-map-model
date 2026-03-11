# Report 1: Data Integrity & Leakage Audit

*Generated: 2026-03-11 06:42*

---

## 1. Artifact Inventory

| Artifact | Exists | Size (MB) | Last Modified |
|----------|--------|-----------|---------------|
| `temporal_model_test_results.npz` | YES | 7.5 | 2026-03-11 06:40 |
| `temporal_model_test_set_with_pred.parquet` | YES | 11.8 | 2026-03-11 06:40 |
| `toronto_temporal_count_model.pkl` | YES | 0.9 | 2026-03-11 06:40 |
| `MODEL_HORSE_RACE_REPORT.md` | YES | 0.0 | 2026-03-03 18:26 |

---

## 2. Temporal Split Summary

> **Note:** Only the test split is available as a saved artifact. Train and validation statistics are estimated from the 60/20/20 temporal split configuration.

| Split | Windows (est.) | Rows (test set actual) | Positive rate |
|-------|---------------|------------------------|---------------|
| Train (est.) | 4,218 | — | — |
| Validation (est.) | 1,406 | — | — |
| **Test** | **1,406** | **311,072** | **1.528%** |

- Test window range: `2021-05-26 13:00` → `2025-03-31 13:00`
- Unique road segments in test: 4,475

The temporal split is implemented by ordering unique `window_start` values and assigning the first 60% to train, next 20% to validation, last 20% to test. This strictly prevents any future crash information from appearing in the training set.

---

## 3. Feature Importance (Permutation)

> `HistGradientBoostingRegressor` does not expose `.feature_importances_`. Permutation importance is computed on a stratified subsample (all crash windows + 5,000 random zero windows) using 10 repeats. Importance = mean increase in MSE when a feature is randomly shuffled.

*Permutation importance computation failed — see logs.*

---

## 4. Feature Leakage Audit

The following columns are explicitly excluded from the feature set in `TemporalCountModelTrainer.prepare_panel_features()`. Each must NOT appear in `feature_columns` used by the trained model.

| Excluded Column | In `feature_columns`? | Status |
|----------------|----------------------|--------|
| `FROM_INTERSECTION_ID` | no | PASS |
| `ROAD_CLASS` | no | PASS |
| `TO_INTERSECTION_ID` | no | PASS |
| `crash_count` | no | PASS |
| `datetime_hour` | no | PASS |
| `day_of_week` | no | PASS |
| `fatalities` | no | PASS |
| `future_crash_count` | no | PASS |
| `future_window_start` | no | PASS |
| `hour_of_day` | no | PASS |
| `is_ksi` | no | PASS |
| `lat_grid` | no | PASS |
| `lon_grid` | no | PASS |
| `month` | no | PASS |
| `sample_weight` | no | PASS |
| `sample_weight_tail` | no | PASS |
| `season` | no | PASS |
| `segment_centroid_lat` | no | PASS |
| `segment_centroid_lon` | no | PASS |
| `segment_id` | no | PASS |
| `window_start` | no | PASS |

**Result: All excluded columns confirmed absent from feature set.**

---

## 5. Target Distribution (Test Set)

- **Zero-crash windows:** 98.47%
- **Mean:** 0.017096
- **Max:** 10
- **p50:** 0  **p90:** 0  **p99:** 1  **p99.9:** 2.0

| Crash count (y) | Rows | % of test |
|-----------------|------|-----------|
| 0 | 306,318 | 98.472% |
| 1 | 4,286 | 1.378% |
| 2 | 406 | 0.131% |
| 3 | 47 | 0.015% |
| 4 | 6 | 0.002% |
| 5 | 6 | 0.002% |
| >5 | 3 | 0.001% |

> **Note:** The sampled training panel uses `negative_multiplier=10`, so the test positive rate (~0.15%) reflects the sampled distribution, not the true hourly sparsity across all Toronto road segments (which is much lower).