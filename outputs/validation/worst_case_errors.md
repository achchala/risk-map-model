# Worst-Case Error Analysis

## Overview
Top-50 predictions with the largest absolute error on the held-out test set.

| Stat | Value |
|---|---|
| Median |y_pred - y_true| | 3.0 |
| Max |y_pred - y_true| | 9.9993 |
| Mean true crash count (top-50) | 3.68 |
| Mean predicted λ (top-50) | 0.0073 |

## Failure Patterns

### road_class
| Category | Count |
|---|---|
| Minor Arterial | 23 |
| Major Arterial | 15 |
| Collector | 7 |
| Local | 3 |
| Trail | 2 |

### weekend_count
13

### weekday_count
37

### freezing_count
6

## Under vs. Over Prediction
- **Under-predicted** (missed crashes): 50/50
- **Over-predicted** (false alarms): 0/50

> Under-prediction (missing real crashes) is the higher-stakes error for a safety routing tool.

## Top-10 Worst Predictions

| segment_id | window_start | future_crash_count | y_pred | _abs_error | ROAD_CLASS |
|---|---|---|---|---|---|
| 912017 | 2023-09-13 13:00:00 | 10.0 | 0.0007 | 9.9993 | Major Arterial |
| 1146226 | 2022-12-18 13:00:00 | 9.0 | 0.0014 | 8.9986 | Collector |
| 1146226 | 2023-06-27 13:00:00 | 6.0 | 0.0904 | 5.9096 | Collector |
| 8089 | 2022-09-01 13:00:00 | 5.0 | 0.0022 | 4.9978 | Major Arterial |
| 14677578 | 2022-10-22 13:00:00 | 5.0 | 0.0033 | 4.9967 | Minor Arterial |
| 1147170 | 2021-07-30 13:00:00 | 5.0 | 0.0046 | 4.9954 | Major Arterial |
| 1147416 | 2024-11-07 13:00:00 | 5.0 | 0.0113 | 4.9887 | Minor Arterial |
| 1147416 | 2024-03-24 13:00:00 | 5.0 | 0.0319 | 4.9681 | Minor Arterial |
| 1147416 | 2024-04-16 13:00:00 | 5.0 | 0.061 | 4.939 | Minor Arterial |
| 1147416 | 2024-07-14 13:00:00 | 4.0 | 0.0 | 4.0 | Minor Arterial |