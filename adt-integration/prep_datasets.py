"""
prep_datasets.py
----------------
Prepares the volume data and crash data as clean, merge-ready datasets.
Each outputs exactly one row per centreline_id.

Run from this directory:
    python prep_datasets.py
"""

import sys
import pandas as pd
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================

VOLUME_PATH = Path(__file__).parent / "svc_most_recent_summary_data.csv"

# Root of the risk-map-model project (for the spatial join pipeline)
RISK_MODEL_ROOT = Path("/Users/adriel.devera/personal/risk-map-model")
DATA_DIR = RISK_MODEL_ROOT / "data"

# Only keep volume counts from years that overlap with crash data
YEAR_MIN = 2006
YEAR_MAX = 2024

# Output files (written to this directory)
VOLUME_OUT = Path(__file__).parent / "volume_clean.csv"
CRASH_OUT  = Path(__file__).parent / "crash_clean.csv"

# =============================================================================
# PART 1 — Volume Data
# =============================================================================

print("=" * 60)
print("PART 1: Volume Data")
print("=" * 60)

vol = pd.read_csv(VOLUME_PATH)
print(f"Loaded {len(vol):,} rows from volume data.")

# Drop rows with missing volume
before = len(vol)
vol = vol.dropna(subset=["avg_daily_vol"])
print(f"Dropped {before - len(vol):,} rows with missing avg_daily_vol. Remaining: {len(vol):,}")

# Parse year and filter to crash data range
vol["count_year"] = pd.to_datetime(vol["latest_count_date_start"], errors="coerce").dt.year
before = len(vol)
vol = vol[vol["count_year"].between(YEAR_MIN, YEAR_MAX)]
print(f"Filtered to {YEAR_MIN}–{YEAR_MAX}: kept {len(vol):,} rows (dropped {before - len(vol):,}).")

# Aggregate: one row per centreline_id (average across multiple counts)
NUMERIC_COLS = [
    "avg_daily_vol",
    "avg_weekday_daily_vol",
    "avg_weekend_daily_vol",
    "avg_wkdy_am_peak_vol",
    "avg_wkdy_pm_peak_vol",
    "avg_speed",
    "avg_85th_percentile_speed",
    "avg_95th_percentile_speed",
    "avg_heavy_pct",
]
agg_cols = [c for c in NUMERIC_COLS if c in vol.columns]

vol_agg = vol.groupby("centreline_id", as_index=False)[agg_cols].mean()
print(f"After aggregation: {len(vol_agg):,} unique centreline_ids.")

# Ensure centreline_id is Int64 to match crash data
vol_agg["centreline_id"] = pd.to_numeric(vol_agg["centreline_id"], errors="coerce").astype("Int64")
null_keys = vol_agg["centreline_id"].isna().sum()
if null_keys:
    print(f"WARNING: {null_keys} rows with unparseable centreline_id — dropping.")
    vol_agg = vol_agg.dropna(subset=["centreline_id"])

print(f"Volume data ready: {len(vol_agg):,} rows | centreline_id dtype = {vol_agg['centreline_id'].dtype}")
print(vol_agg.head(3).to_string(index=False))

# =============================================================================
# PART 2 — Crash Data  (produced via spatial join from the risk-map-model pipeline)
# =============================================================================

print()
print("=" * 60)
print("PART 2: Crash Data  (spatial join)")
print("=" * 60)

sys.path.insert(0, str(RISK_MODEL_ROOT))
from src.data_processing.data_loader import load_and_clean_data
from src.data_processing.spatial_join_fast import perform_spatial_join_fast

print("Running spatial join — this may take a minute...")
collision_data, ksi_data, road_network = load_and_clean_data(DATA_DIR)
segment_crashes = perform_spatial_join_fast(collision_data, ksi_data, road_network)

print(f"Spatial join produced {len(segment_crashes):,} segments.")

# Keep only the columns we need
crash = segment_crashes[["CENTRELINE_ID", "segment_length", "num_total_crashes"]].copy()
crash = crash.rename(columns={
    "CENTRELINE_ID":    "centreline_id",
    "num_total_crashes": "crash_count",
})

# Each row is already one segment, but assert uniqueness
dupes = crash["centreline_id"].duplicated().sum()
if dupes > 0:
    print(f"WARNING: {dupes} duplicate centreline_ids found — aggregating.")
    crash = crash.groupby("centreline_id", as_index=False).agg(
        crash_count=("crash_count", "sum"),
        segment_length=("segment_length", "first"),
    )
else:
    print("One row per centreline_id confirmed. No aggregation needed.")

# Clean segment_length
crash["segment_length"] = pd.to_numeric(crash["segment_length"], errors="coerce")
bad_lengths = crash["segment_length"].isna().sum()
if bad_lengths:
    print(f"WARNING: {bad_lengths} rows with unparseable segment_length — dropping.")
    crash = crash.dropna(subset=["segment_length"])

zero_lengths = (crash["segment_length"] <= 0).sum()
if zero_lengths:
    print(f"WARNING: {zero_lengths} rows with zero/negative segment_length — dropping.")
    crash = crash[crash["segment_length"] > 0]

# Ensure centreline_id is Int64
crash["centreline_id"] = pd.to_numeric(crash["centreline_id"], errors="coerce").astype("Int64")
null_keys = crash["centreline_id"].isna().sum()
if null_keys:
    print(f"WARNING: {null_keys} rows with unparseable centreline_id — dropping.")
    crash = crash.dropna(subset=["centreline_id"])

print(f"Crash data ready: {len(crash):,} rows | centreline_id dtype = {crash['centreline_id'].dtype}")
print(crash.head(3).to_string(index=False))

# =============================================================================
# PART 3 — Overlap check
# =============================================================================

print()
print("=" * 60)
print("PART 3: Overlap Check")
print("=" * 60)

vol_ids   = set(vol_agg["centreline_id"].dropna())
crash_ids = set(crash["centreline_id"].dropna())
overlap   = vol_ids & crash_ids

print(f"Unique IDs in volume data  : {len(vol_ids):,}")
print(f"Unique IDs in crash data   : {len(crash_ids):,}")
print(f"Overlapping (will merge)   : {len(overlap):,}")
print(f"Crash IDs with no volume   : {len(crash_ids - vol_ids):,}")
print(f"Volume IDs with no crash   : {len(vol_ids - crash_ids):,}")

# =============================================================================
# PART 4 — Save
# =============================================================================

vol_agg.to_csv(VOLUME_OUT, index=False)
crash.to_csv(CRASH_OUT, index=False)

print()
print(f"Saved volume data  → {VOLUME_OUT}")
print(f"Saved crash data   → {CRASH_OUT}")
print("Both datasets ready to merge on centreline_id.")
