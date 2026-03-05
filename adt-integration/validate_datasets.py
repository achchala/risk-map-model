"""
validate_datasets.py
--------------------
Proves volume_clean.csv and crash_clean.csv are structurally
identical on their join key before any merge or modelling.

Checks:
  1. Key dtype match
  2. Uniqueness in both datasets (no fan-out trap)
  3. Volume math boundaries (no <= 0 values that break log())

Run from this directory:
    python validate_datasets.py
"""

import sys
import pandas as pd

VOLUME_PATH = "volume_clean.csv"
CRASH_PATH  = "crash_clean.csv"
VOLUME_COL  = "avg_daily_vol"

PASS = "[PASS]"
FAIL = "[FAIL]"
WARN = "[WARN]"

errors = []

# ── Load ──────────────────────────────────────────────────────────────────────
vol   = pd.read_csv(VOLUME_PATH)
crash = pd.read_csv(CRASH_PATH)
print(f"Loaded volume : {len(vol):>7,} rows")
print(f"Loaded crash  : {len(crash):>7,} rows")
print()

# =============================================================================
# CHECK 1 — Key dtype match
# =============================================================================
print("=" * 60)
print("CHECK 1: centreline_id dtype match")
print("=" * 60)

vol_dtype   = vol["centreline_id"].dtype
crash_dtype = crash["centreline_id"].dtype

print(f"  volume dtype : {vol_dtype}")
print(f"  crash dtype  : {crash_dtype}")

if vol_dtype == crash_dtype:
    print(f"  {PASS} Types match — merge will align correctly.")
else:
    # Attempt to fix by casting both to int64
    print(f"  {FAIL} Type mismatch — forcing both to int64.")
    vol["centreline_id"]   = pd.to_numeric(vol["centreline_id"],   errors="coerce").astype("int64")
    crash["centreline_id"] = pd.to_numeric(crash["centreline_id"], errors="coerce").astype("int64")
    if vol["centreline_id"].dtype == crash["centreline_id"].dtype:
        print(f"  {PASS} Both coerced to int64 successfully.")
    else:
        msg = "Could not align centreline_id types."
        print(f"  {FAIL} {msg}")
        errors.append(msg)

# =============================================================================
# CHECK 2 — Uniqueness (fan-out trap)
# =============================================================================
print()
print("=" * 60)
print("CHECK 2: Uniqueness — no duplicate centreline_ids")
print("=" * 60)

for label, df in [("volume", vol), ("crash", crash)]:
    dupes = df["centreline_id"].duplicated()
    n_dupes = dupes.sum()
    if n_dupes == 0:
        print(f"  {PASS} {label:8s} — all {len(df):,} IDs are unique.")
    else:
        dup_ids = df.loc[dupes, "centreline_id"].unique()
        print(f"  {FAIL} {label:8s} — {n_dupes:,} duplicate rows across {len(dup_ids):,} IDs.")
        print(f"         Sample duplicate IDs: {dup_ids[:5].tolist()}")
        errors.append(f"{label}: {n_dupes} duplicate centreline_id rows.")

# =============================================================================
# CHECK 3 — Math boundaries (log-safe volume)
# =============================================================================
print()
print("=" * 60)
print("CHECK 3: Volume math boundaries  (no value <= 0 before log)")
print("=" * 60)

# 3a. Nulls
nulls = vol[VOLUME_COL].isna()
n_nulls = nulls.sum()
if n_nulls == 0:
    print(f"  {PASS} No null values in {VOLUME_COL}.")
else:
    print(f"  {WARN} {n_nulls:,} null values — dropping before log check.")
    vol = vol[~nulls]

# 3b. Zero or negative
bad_mask = vol[VOLUME_COL] <= 0
n_bad    = bad_mask.sum()
if n_bad == 0:
    print(f"  {PASS} All {VOLUME_COL} values are > 0 — safe for log().")
else:
    print(f"  {FAIL} {n_bad:,} rows with {VOLUME_COL} <= 0 (log-undefined):")
    print(vol[bad_mask][["centreline_id", VOLUME_COL]].to_string(index=False))
    vol = vol[~bad_mask]
    print(f"  Dropped {n_bad:,} rows. Remaining: {len(vol):,}")
    errors.append(f"{n_bad} rows had {VOLUME_COL} <= 0 and were dropped.")

# 3c. Descriptive stats
print()
print(f"  {VOLUME_COL} summary after cleaning:")
stats = vol[VOLUME_COL].describe(percentiles=[.01, .25, .5, .75, .99])
for stat, val in stats.items():
    print(f"    {stat:5s} : {val:>10,.1f}")

# =============================================================================
# SUMMARY
# =============================================================================
print()
print("=" * 60)
print("VALIDATION SUMMARY")
print("=" * 60)
if not errors:
    print(f"  {PASS} All checks passed. Datasets are merge-ready.")
else:
    print(f"  {FAIL} {len(errors)} issue(s) found:")
    for e in errors:
        print(f"       • {e}")
    sys.exit(1)

# Quick merge smoke-test
merged = vol.merge(crash, on="centreline_id", how="inner")
print(f"  Smoke-test inner merge → {len(merged):,} rows  "
      f"({len(merged)/len(crash)*100:.1f}% of crash segments have volume data)")
