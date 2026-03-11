"""
Top Risk Drivers — Correlation Bar Chart
Pearson correlation of each feature with crash_count, sorted by absolute value.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

CSV_PATH = (
    "/Users/adriel.devera/Desktop/"
    "Traffic Volumes - Midblock Vehicle Speed, Volume and Classification Counts/"
    "model_dataset.csv"
)

df = pd.read_csv(CSV_PATH)

# Drop non-numeric and identifier columns
drop_cols = ["centreline_id", "crash_rate"]  # crash_rate is derived from crash_count
numeric = df.drop(columns=drop_cols, errors="ignore").select_dtypes(include="number")

corr = numeric.corr()["crash_count"].drop("crash_count").sort_values(key=abs, ascending=False)

fig, ax = plt.subplots(figsize=(9, 6))
colors = ["#d62728" if v > 0 else "#1f77b4" for v in corr]
sns.barplot(x=corr.values, y=corr.index, palette=colors, ax=ax)

ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("Pearson Correlation with crash_count")
ax.set_title("Top Risk Drivers\n(correlation with crash_count)", fontsize=13)
ax.set_ylabel("")

plt.tight_layout()
plt.savefig("top_risk_drivers.png", dpi=150)
plt.show()
print("Saved: top_risk_drivers.png")
