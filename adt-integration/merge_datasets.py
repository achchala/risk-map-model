"""
merge_datasets.py
-----------------
Inner join of volume_clean.csv and crash_clean.csv on centreline_id.
Produces model_dataset.csv — the single source of truth for all modelling.

Run from this directory:
    python merge_datasets.py
"""

import numpy as np
import pandas as pd

VOLUME_PATH = "volume_clean.csv"
CRASH_PATH  = "crash_clean.csv"
OUTPUT_PATH = "model_dataset.csv"

# ── Load ──────────────────────────────────────────────────────────────────────
vol   = pd.read_csv(VOLUME_PATH)
crash = pd.read_csv(CRASH_PATH)

print(f"Volume rows  : {len(vol):,}")
print(f"Crash rows   : {len(crash):,}")

# ── Merge ─────────────────────────────────────────────────────────────────────
merged = crash.merge(vol, on="centreline_id", how="inner")

print(f"\nInner join result : {len(merged):,} rows")
print(f"  Crash segments that gained volume data : {len(merged):,} / {len(crash):,}  "
      f"({len(merged)/len(crash)*100:.1f}%)")
print(f"  Volume segments matched to crash data  : {len(merged):,} / {len(vol):,}  "
      f"({len(merged)/len(vol)*100:.1f}%)")

# ── Derived columns ───────────────────────────────────────────────────────────
# log_volume: the feature that goes into the regression offset / exposure term
merged["log_volume"] = np.log(merged["avg_daily_vol"])

# crash_rate: crashes per km  (segment_length is in metres)
merged["crash_rate"] = merged["crash_count"] / (merged["segment_length"] / 1000)

# ── Sanity checks ─────────────────────────────────────────────────────────────
assert merged["centreline_id"].is_unique, "Duplicate centreline_ids after merge!"
assert merged["log_volume"].isna().sum() == 0, "Nulls in log_volume!"
assert (merged["segment_length"] > 0).all(), "Zero/negative segment_length!"

print(f"\nDerived columns added:")
print(f"  log_volume  — range [{merged['log_volume'].min():.2f}, {merged['log_volume'].max():.2f}]")
print(f"  crash_rate  — range [{merged['crash_rate'].min():.2f}, {merged['crash_rate'].max():.2f}] crashes/km")

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\nFinal dataset shape : {merged.shape}")
print(f"Columns : {list(merged.columns)}")
print()
print(merged.describe(percentiles=[.25, .5, .75]).T.to_string())

# ── Exposure columns ──────────────────────────────────────────────────────────
merged["exposure"]     = merged["avg_daily_vol"] * merged["segment_length"]
merged["log_exposure"] = np.log(merged["exposure"])

# ── Save ──────────────────────────────────────────────────────────────────────
merged.to_csv(OUTPUT_PATH, index=False)
print(f"\nSaved → {OUTPUT_PATH}")
