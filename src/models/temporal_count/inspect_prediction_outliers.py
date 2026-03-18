#!/usr/bin/env python3
"""
Inspect rows with the highest predicted λ to find pathological features or bugs.

Loads the test set with predictions saved during training, sorts by y_pred descending,
and prints the top N rows with segment_id, window_start, future_crash_count (y_test),
y_pred, and all feature columns (to spot huge lags, rolling stats, duplicates, etc.).

Run from project root after training:
  python train_temporal_model.py   # saves temporal_model_test_set_with_pred.parquet
  python inspect_prediction_outliers.py [N]
  Default N=50.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Project root
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import OUTPUTS_DIR


def main() -> None:
    test_set_path = OUTPUTS_DIR / "reports" / "temporal_model_test_set_with_pred.parquet"
    if not test_set_path.exists():
        print(f"Test set not found: {test_set_path}")
        print("Run training first: python train_temporal_model.py")
        sys.exit(1)

    n_top = 50
    if len(sys.argv) > 1:
        try:
            n_top = int(sys.argv[1])
        except ValueError:
            pass

    print(f"Loading test set from {test_set_path}...")
    df = pd.read_parquet(test_set_path)
    if "y_pred" not in df.columns:
        print("No 'y_pred' column in test set. Re-run training to save test_data_with_pred.")
        sys.exit(1)

    df = df.sort_values("y_pred", ascending=False).reset_index(drop=True)
    top = df.head(n_top)

    id_cols = ["segment_id", "window_start", "future_crash_count", "y_pred"]
    feature_cols = [c for c in df.columns if c not in id_cols and c not in {"sample_weight", "future_window_start", "crash_count", "datetime_hour", "lat_grid", "lon_grid"}]

    print(f"\nTop {n_top} rows by y_pred (highest predictions first)")
    print("=" * 80)
    print(top[id_cols].to_string())
    print("\nFeature values for top rows (look for huge values, NaN, duplicates)")
    print("=" * 80)
    print(top[feature_cols].to_string())

    # Duplicate check: same (segment_id, window_start) repeated?
    dup = df.groupby(["segment_id", "window_start"]).size()
    dup = dup[dup > 1]
    if len(dup) > 0:
        print(f"\nWARNING: {len(dup)} (segment_id, window_start) pairs appear more than once (duplicate merges?).")
    else:
        print("\nNo duplicate (segment_id, window_start) in test set.")

    print("\nDone. Use this to find bad feature values (e.g. rolling_max_30d = 10000) or join bugs.")


if __name__ == "__main__":
    main()
