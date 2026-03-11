"""
Data Quality / Gaps — Missingness Matrix
Uses missingno to show which columns are missing and whether gaps co-occur.
Install: pip install missingno
"""

import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt

CSV_PATH = (
    "/Users/adriel.devera/Desktop/"
    "Traffic Volumes - Midblock Vehicle Speed, Volume and Classification Counts/"
    "model_dataset.csv"
)

df = pd.read_csv(CSV_PATH)

# --- 1. Matrix: row-level missingness pattern ---
fig, ax = plt.subplots(figsize=(12, 6))
msno.matrix(df, ax=ax, sparkline=False, fontsize=10, color=(0.25, 0.45, 0.75))
ax.set_title("Missingness Matrix — one row per road segment", fontsize=13, pad=12)
plt.tight_layout()
plt.savefig("data_quality_matrix.png", dpi=150)
plt.show()
print("Saved: data_quality_matrix.png")

# --- 2. Bar: % present per column ---
fig2, ax2 = plt.subplots(figsize=(10, 5))
msno.bar(df, ax=ax2, fontsize=10, color="#4c72b0")
ax2.set_title("Data Completeness by Column", fontsize=13, pad=12)
plt.tight_layout()
plt.savefig("data_quality_bar.png", dpi=150)
plt.show()
print("Saved: data_quality_bar.png")

# --- 3. Print summary ---
missing = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
missing = missing[missing > 0]
print("\nMissing-value summary (% of rows):")
print(missing.to_string())
