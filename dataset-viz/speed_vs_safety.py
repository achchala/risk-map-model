"""
Speed vs. Safety — Split Violin Plot
Compares distributions of avg_speed, avg_85th_percentile_speed, and
avg_95th_percentile_speed for segments with 0 crashes vs 1+ crashes.
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

speed_cols = ["avg_speed", "avg_85th_percentile_speed", "avg_95th_percentile_speed"]
df["crash_group"] = df["crash_count"].apply(lambda x: "1+ crashes" if x > 0 else "0 crashes")

# Melt to long format for faceting
long = df[speed_cols + ["crash_group"]].dropna().melt(
    id_vars="crash_group",
    var_name="speed_metric",
    value_name="speed_kmh",
)

# Friendly labels
label_map = {
    "avg_speed": "Mean Speed",
    "avg_85th_percentile_speed": "85th Pct Speed",
    "avg_95th_percentile_speed": "95th Pct Speed",
}
long["speed_metric"] = long["speed_metric"].map(label_map)

fig, ax = plt.subplots(figsize=(11, 6))

sns.violinplot(
    data=long,
    x="speed_metric",
    y="speed_kmh",
    hue="crash_group",
    split=True,
    inner="quartile",
    palette={"0 crashes": "#4c72b0", "1+ crashes": "#d62728"},
    ax=ax,
    linewidth=0.8,
)

ax.set_xlabel("")
ax.set_ylabel("Speed (km/h)")
ax.set_title("Speed Distribution: 0-crash vs 1+-crash Segments", fontsize=13)
ax.legend(title="")
sns.despine()

plt.tight_layout()
plt.savefig("speed_vs_safety.png", dpi=150)
plt.show()
print("Saved: speed_vs_safety.png")
