"""
Road Type Clusters — 2D Density Hexbin
avg_daily_vol vs avg_speed to surface implicit road-type clusters
(arterials = high volume + high speed, locals = low volume + low speed).
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

CSV_PATH = (
    "/Users/adriel.devera/Desktop/"
    "Traffic Volumes - Midblock Vehicle Speed, Volume and Classification Counts/"
    "model_dataset.csv"
)

df = pd.read_csv(CSV_PATH)

# Only rows with both volume and speed
sub = df.dropna(subset=["avg_daily_vol", "avg_speed"])

fig, ax = plt.subplots(figsize=(9, 7))

hb = ax.hexbin(
    sub["avg_daily_vol"],
    sub["avg_speed"],
    gridsize=55,
    cmap="YlOrRd",
    mincnt=1,
    norm=mcolors.LogNorm(),
)

cb = fig.colorbar(hb, ax=ax, label="Segment count (log scale)")

# Loose cluster annotations
ax.annotate("Local streets\n(low vol, low speed)", xy=(500, 20), fontsize=9,
            color="#333", ha="center",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))
ax.annotate("Arterials\n(high vol, moderate speed)", xy=(18000, 43), fontsize=9,
            color="#333", ha="center",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))
ax.annotate("High-speed corridors", xy=(8000, 58), fontsize=9,
            color="#333", ha="center",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

ax.set_xlabel("Average Daily Volume (vehicles/day)")
ax.set_ylabel("Average Speed (km/h)")
ax.set_title("Road Type Clusters\n(avg_daily_vol vs avg_speed)", fontsize=13)

plt.tight_layout()
plt.savefig("road_type_clusters.png", dpi=150)
plt.show()
print("Saved: road_type_clusters.png")
