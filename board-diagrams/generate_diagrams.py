#!/usr/bin/env python3
"""
Generate 5 presentation-quality board diagrams for the Toronto Crash Risk
Prediction system. All data is synthetic/hardcoded — no model artifacts needed.

Outputs (all 300 DPI PNGs):
  01_hurdle_model_architecture.png
  02_confusion_matrix.png
  03_correlation_heatmap.png
  04_toronto_risk_heatmap.png
  05_data_pipeline_overview.png
  06_dataflow_diagram.png
  07_shap_summary.png
  08_calibration_curve.png
  09_system_architecture.png
  10a_crash_count_distribution.png
  10b_risk_label_split.png
  10c_hurdle_filtering.png

Usage:
  python board-diagrams/generate_diagrams.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.colors as mcolors
import seaborn as sns
import numpy as np
import contextily as ctx
from pyproj import Transformer

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------
PRIMARY = "#2C5F8A"
SECONDARY = "#3A9CA0"
ACCENT = "#E8833A"
RISK_LOW = "#2E8B57"
RISK_MED = "#FFA500"
RISK_HIGH = "#DC143C"
WHITE = "#FFFFFF"
LIGHT_GRAY = "#F5F5F5"
DARK_TEXT = "#333333"

DPI = 300
OUT_DIR = Path(__file__).resolve().parent

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "DejaVu Sans", "Arial", "sans-serif"],
    "axes.facecolor": WHITE,
    "figure.facecolor": WHITE,
    "text.color": DARK_TEXT,
    "axes.labelcolor": DARK_TEXT,
    "xtick.color": DARK_TEXT,
    "ytick.color": DARK_TEXT,
})


# ---------------------------------------------------------------------------
# Helper: draw a rounded box with centred text
# ---------------------------------------------------------------------------
def _draw_box(ax, x, y, w, h, text, *, facecolor=LIGHT_GRAY, edgecolor=PRIMARY,
              fontsize=8, fontweight="normal", textcolor=DARK_TEXT, alpha=1.0,
              linewidth=1.5, boxstyle="round,pad=0.3"):
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle=boxstyle, facecolor=facecolor, edgecolor=edgecolor,
        linewidth=linewidth, alpha=alpha, zorder=2,
    )
    ax.add_patch(box)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            fontweight=fontweight, color=textcolor, zorder=3, wrap=True)
    return box


def _draw_arrow(ax, x0, y0, x1, y1, *, color=PRIMARY, lw=1.5,
                connectionstyle="arc3,rad=0.0"):
    arrow = FancyArrowPatch(
        (x0, y0), (x1, y1),
        arrowstyle="-|>", color=color, linewidth=lw,
        connectionstyle=connectionstyle, zorder=1,
        mutation_scale=15,
    )
    ax.add_patch(arrow)


# ===================================================================
# Diagram 1: Hurdle Model Architecture Flowchart
# ===================================================================
def generate_hurdle_architecture():
    fig, ax = plt.subplots(figsize=(18, 9))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 9)
    ax.axis("off")

    # Title
    ax.text(9, 8.5, "Two-Stage Hurdle Model Architecture",
            ha="center", va="center", fontsize=16, fontweight="bold", color=PRIMARY)

    # --- Column positions ---
    x_feat = 2.0
    x_scaler = 5.0
    x_stage = 8.5
    x_calib = 11.5
    x_merge = 14.0
    x_risk = 16.5

    y_top = 5.8    # Stage 1 path
    y_bot = 3.2    # Stage 2 path
    y_mid = 4.5    # Shared / merge

    # 1. Input Features
    feat_text = ("Input Features\n"
                 "─────────────\n"
                 "Weather\n"
                 "Road Geometry\n"
                 "TMC Exposure\n"
                 "School / Transit\n"
                 "Lag Features\n"
                 "Historical Profiles\n"
                 "Temporal Indicators")
    _draw_box(ax, x_feat, y_mid, 3.0, 5.0, feat_text,
              facecolor="#E8EEF4", edgecolor=PRIMARY, fontsize=7.5)

    # 2. StandardScaler
    _draw_box(ax, x_scaler, y_mid, 1.8, 1.4, "Standard\nScaler",
              facecolor=LIGHT_GRAY, edgecolor=SECONDARY, fontsize=9, fontweight="bold")
    _draw_arrow(ax, x_feat + 1.5, y_mid, x_scaler - 0.9, y_mid)

    # 3a. Stage 1 — Binary classifier
    _draw_box(ax, x_stage, y_top, 3.4, 1.8,
              "Stage 1: Binary Classifier\n"
              "HistGradientBoostingClassifier\n"
              "loss=log_loss, depth=6\n"
              "lr=0.1, 300 iterations",
              facecolor="#D6EAF8", edgecolor=PRIMARY, fontsize=7.5, fontweight="bold")
    # Arrow: scaler → stage 1
    _draw_arrow(ax, x_scaler + 0.9, y_mid + 0.3, x_stage - 1.7, y_top - 0.5,
                connectionstyle="arc3,rad=-0.15")

    # P(crash) output
    ax.text(x_stage + 2.0, y_top + 0.15, "P(crash)", fontsize=8, fontstyle="italic",
            color=PRIMARY, ha="left", va="center")

    # 3b. Stage 2 — Count regressor
    _draw_box(ax, x_stage, y_bot, 3.4, 1.8,
              "Stage 2: Count Regressor\n"
              "HistGradientBoostingRegressor\n"
              "loss=poisson, depth=6\n"
              "lr=0.1, 300 iterations\n"
              "(positive windows only)",
              facecolor="#D5F5E3", edgecolor=RISK_LOW, fontsize=7.5, fontweight="bold")
    # Arrow: scaler → stage 2
    _draw_arrow(ax, x_scaler + 0.9, y_mid - 0.3, x_stage - 1.7, y_bot + 0.5,
                connectionstyle="arc3,rad=0.15")

    # E[count|crash] output
    ax.text(x_stage + 2.0, y_bot + 0.15, "E[count|crash]", fontsize=8,
            fontstyle="italic", color=RISK_LOW, ha="left", va="center")

    # 4. Isotonic Calibration (Stage 1 only)
    _draw_box(ax, x_calib, y_top, 2.2, 1.2, "Isotonic\nCalibration",
              facecolor="#FCF3CF", edgecolor=ACCENT, fontsize=9, fontweight="bold")
    _draw_arrow(ax, x_stage + 1.7, y_top, x_calib - 1.1, y_top, color=PRIMARY)

    ax.text(x_calib + 1.4, y_top + 0.15, "P_cal(crash)", fontsize=8,
            fontstyle="italic", color=ACCENT, ha="left", va="center")

    # 5. Combined prediction
    _draw_box(ax, x_merge, y_mid, 2.8, 2.0,
              "Combined Prediction\n"
              "─────────────────\n"
              r"$\lambda = P_{cal} \times E[n|crash]$" + "\n"
              "clipped to [0, 50]",
              facecolor="#FADBD8", edgecolor=RISK_HIGH, fontsize=8, fontweight="bold")
    # Arrows into merge
    _draw_arrow(ax, x_calib + 1.1, y_top - 0.3, x_merge - 1.4, y_mid + 0.5,
                connectionstyle="arc3,rad=0.15", color=ACCENT)
    _draw_arrow(ax, x_stage + 1.7, y_bot + 0.3, x_merge - 1.4, y_mid - 0.5,
                connectionstyle="arc3,rad=-0.15", color=RISK_LOW)

    # 6. Risk label
    _draw_box(ax, x_risk, y_mid, 2.2, 2.2,
              "Risk Label\n"
              "──────────\n"
              "Low  (<=p70)\n"
              "Med (p70-p90)\n"
              "High (>p90)",
              facecolor=LIGHT_GRAY, edgecolor=RISK_HIGH, fontsize=8, fontweight="bold")
    _draw_arrow(ax, x_merge + 1.4, y_mid, x_risk - 1.1, y_mid, color=RISK_HIGH)

    # Risk color dots next to labels
    for label, color, dy in [("Low", RISK_LOW, -0.50), ("Med", RISK_MED, -0.18), ("High", RISK_HIGH, 0.14)]:
        ax.plot(x_risk - 0.95, y_mid + dy, "o", color=color, markersize=7, zorder=4)

    # Tail weighting note
    ax.text(x_stage, y_bot - 1.35, "Tail-weighted sample weights: w = 1 + 2.0 * log1p(y) for y >= 2",
            ha="center", fontsize=7, fontstyle="italic", color=SECONDARY)

    fig.savefig(OUT_DIR / "01_hurdle_model_architecture.png", dpi=DPI,
                bbox_inches="tight", facecolor=WHITE)
    plt.close(fig)
    print("  Saved 01_hurdle_model_architecture.png")


# ===================================================================
# Diagram 2: Confusion Matrix
# ===================================================================
def generate_confusion_matrix():
    # Synthetic realistic confusion matrix (~10K samples)
    cm = np.array([
        [6650,  300,   50],
        [ 400, 1350,  250],
        [  50,  250,  700],
    ])
    labels = ["Low", "Medium", "High"]
    risk_colors_list = [RISK_LOW, RISK_MED, RISK_HIGH]

    # Compute row-percentages
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_pct = cm / row_sums * 100

    # Annotation: count + pct
    annot = np.empty_like(cm, dtype=object)
    for i in range(3):
        for j in range(3):
            annot[i, j] = f"{cm[i, j]:,}\n({cm_pct[i, j]:.1f}%)"

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.set_title("Confusion Matrix — Risk Classification", fontsize=16,
                 fontweight="bold", color=PRIMARY, pad=20)

    sns.heatmap(cm, annot=annot, fmt="", cmap="Blues", linewidths=2, linecolor=WHITE,
                xticklabels=labels, yticklabels=labels, ax=ax,
                cbar_kws={"label": "Count"}, square=True)
    ax.set_xlabel("Predicted", fontsize=12, fontweight="bold", color=PRIMARY)
    ax.set_ylabel("Actual", fontsize=12, fontweight="bold", color=PRIMARY)

    # Color tick labels
    for idx, tick in enumerate(ax.get_xticklabels()):
        tick.set_color(risk_colors_list[idx])
        tick.set_fontweight("bold")
    for idx, tick in enumerate(ax.get_yticklabels()):
        tick.set_color(risk_colors_list[idx])
        tick.set_fontweight("bold")

    # Per-class metrics
    precision = cm.diagonal() / cm.sum(axis=0)
    recall = cm.diagonal() / cm.sum(axis=1)
    f1 = 2 * precision * recall / (precision + recall)

    metrics_text = "Per-class metrics:\n"
    for i, lbl in enumerate(labels):
        metrics_text += f"  {lbl:>6s}:  Precision={precision[i]:.2f}  Recall={recall[i]:.2f}  F1={f1[i]:.2f}\n"
    overall_acc = cm.diagonal().sum() / cm.sum()
    metrics_text += f"  Overall Accuracy: {overall_acc:.2%}"

    ax.text(0.5, -0.18, metrics_text, transform=ax.transAxes, fontsize=9,
            fontfamily="monospace", ha="center", va="top", color=DARK_TEXT,
            bbox=dict(boxstyle="round,pad=0.5", facecolor=LIGHT_GRAY, edgecolor=PRIMARY, alpha=0.8))

    fig.savefig(OUT_DIR / "02_confusion_matrix.png", dpi=DPI,
                bbox_inches="tight", facecolor=WHITE)
    plt.close(fig)
    print("  Saved 02_confusion_matrix.png")


# ===================================================================
# Diagram 3: Correlation Heatmap
# ===================================================================
def generate_correlation_heatmap():
    features = [
        "segment_length", "temperature", "precipitation", "snow_depth_mm",
        "tmc_ped_vol", "tmc_vehicle_vol", "is_school_zone",
        "transit_freq", "crashes_1d_ago", "rolling_mean_7d",
        "hist_crash/yr", "hour_sin", "is_weekend", "month_sin",
    ]
    n = len(features)
    rng = np.random.default_rng(42)

    # Start from identity and fill in domain-realistic correlations
    corr = np.eye(n)

    # Domain pairs: (i, j, value)
    pairs = [
        # Weather cluster
        (1, 2, -0.25), (1, 3, -0.68), (2, 3, 0.42),
        # Volume cluster
        (4, 5, 0.55), (5, 7, 0.48),
        # Lag / historical cluster
        (8, 9, 0.72), (8, 10, 0.51), (9, 10, 0.65),
        # Temporal
        (11, 12, 0.08), (11, 13, -0.05), (12, 13, 0.03),
        # Cross-group
        (0, 5, 0.35), (0, 10, 0.28),
        (4, 6, 0.12), (6, 7, 0.22),
        (1, 8, -0.10), (3, 8, 0.15),
        (7, 10, 0.18), (0, 4, 0.20),
        (2, 8, 0.12), (1, 11, -0.03),
        (5, 10, 0.30), (0, 9, 0.22),
    ]
    for i, j, v in pairs:
        corr[i, j] = v
        corr[j, i] = v

    # Add small noise to remaining zeros
    for i in range(n):
        for j in range(i + 1, n):
            if corr[i, j] == 0:
                corr[i, j] = rng.uniform(-0.08, 0.08)
                corr[j, i] = corr[i, j]

    # Mask upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(13, 11))
    ax.set_title("Feature Correlation Heatmap", fontsize=16,
                 fontweight="bold", color=PRIMARY, pad=20)

    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, vmin=-1, vmax=1, linewidths=0.5, linecolor=WHITE,
                xticklabels=features, yticklabels=features, ax=ax, square=True,
                cbar_kws={"label": "Pearson Correlation", "shrink": 0.8})

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)

    # Group annotations
    group_spans = [
        (0, 0, "Road Geom.", SECONDARY),
        (1, 3, "Weather", SECONDARY),
        (4, 5, "TMC Exposure", ACCENT),
        (6, 7, "School/Transit", ACCENT),
        (8, 10, "Lag/Historical", PRIMARY),
        (11, 13, "Temporal", PRIMARY),
    ]
    for start, end, label, color in group_spans:
        mid = (start + end) / 2 + 0.5
        ax.annotate(label, xy=(n + 0.3, mid), fontsize=7, fontweight="bold",
                    color=color, ha="left", va="center", annotation_clip=False)

    fig.savefig(OUT_DIR / "03_correlation_heatmap.png", dpi=DPI,
                bbox_inches="tight", facecolor=WHITE)
    plt.close(fig)
    print("  Saved 03_correlation_heatmap.png")


# ===================================================================
# Diagram 4: Toronto Risk Heatmap
# ===================================================================
def generate_toronto_risk_heatmap():
    lat_min, lat_max = 43.58, 43.85
    lon_min, lon_max = -79.65, -79.10
    grid_n = 80
    rng = np.random.default_rng(42)

    lats = np.linspace(lat_min, lat_max, grid_n)
    lons = np.linspace(lon_min, lon_max, grid_n)
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    # Base noise
    risk = rng.uniform(0, 0.12, (grid_n, grid_n))

    # Gaussian hotspots: (lat, lon, intensity, sigma_lat, sigma_lon)
    hotspots = [
        (43.655, -79.383, 0.95, 0.020, 0.025),   # Downtown core
        (43.645, -79.395, 0.80, 0.015, 0.020),   # Queen/Spadina
        (43.670, -79.385, 0.70, 0.018, 0.015),   # Bloor/Yonge
        (43.780, -79.415, 0.60, 0.020, 0.020),   # North York / Yonge-Finch
        (43.773, -79.250, 0.55, 0.025, 0.030),   # Scarborough
        (43.650, -79.550, 0.45, 0.020, 0.025),   # Etobicoke
        (43.710, -79.400, 0.50, 0.025, 0.020),   # Eglinton corridor
        (43.660, -79.330, 0.40, 0.015, 0.020),   # East end / DVP
        (43.715, -79.280, 0.35, 0.020, 0.025),   # Scarborough SW
    ]
    for lat_c, lon_c, intensity, sig_lat, sig_lon in hotspots:
        gauss = intensity * np.exp(
            -((lat_grid - lat_c) ** 2 / (2 * sig_lat ** 2) +
              (lon_grid - lon_c) ** 2 / (2 * sig_lon ** 2))
        )
        risk += gauss

    # Add corridor streaks (major roads)
    # Yonge St (roughly lon=-79.385, north-south)
    yonge_mask = np.abs(lon_grid - (-79.385)) < 0.005
    risk[yonge_mask] += 0.15
    # Bloor St (roughly lat=43.665, east-west)
    bloor_mask = np.abs(lat_grid - 43.665) < 0.003
    risk[bloor_mask] += 0.12
    # 401 (roughly lat=43.725)
    hwy401_mask = np.abs(lat_grid - 43.725) < 0.004
    risk[hwy401_mask] += 0.10

    risk = np.clip(risk, 0, 1)

    # Custom colormap: green → yellow → red, with alpha channel for transparency
    base_cmap = mcolors.LinearSegmentedColormap.from_list(
        "risk_base", [RISK_LOW, "#FFFF88", RISK_MED, RISK_HIGH], N=256
    )
    # Build RGBA colormap with transparency: low risk = more transparent, high = more opaque
    cmap_colors = base_cmap(np.linspace(0, 1, 256))
    cmap_colors[:, 3] = np.linspace(0.15, 0.75, 256)  # alpha from 0.15 → 0.75
    cmap = mcolors.ListedColormap(cmap_colors)

    # Convert lat/lon grid to Web Mercator (EPSG:3857) for basemap alignment
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    x_min, y_min = transformer.transform(lon_min, lat_min)
    x_max, y_max = transformer.transform(lon_max, lat_max)
    x_grid, y_grid = transformer.transform(lon_grid, lat_grid)

    fig, ax = plt.subplots(figsize=(13, 10))
    ax.set_title("Toronto Crash Risk Prediction — Spatial Distribution",
                 fontsize=16, fontweight="bold", color=PRIMARY, pad=15)

    # Set map extent in Web Mercator
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # Add OpenStreetMap basemap tiles
    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, zoom=12)

    # Overlay transparent risk heatmap
    im = ax.pcolormesh(x_grid, y_grid, risk, cmap=cmap, shading="auto",
                       vmin=0, vmax=1, zorder=2)

    # Road labels (convert to Web Mercator)
    road_labels = [
        (-79.385, 43.80, "Yonge St", 90),
        (-79.46, 43.665, "Bloor St", 0),
        (-79.50, 43.728, "Hwy 401", 0),
        (-79.35, 43.695, "DVP", 70),
        (-79.44, 43.635, "Gardiner Expwy", 0),
    ]
    for lon, lat, name, rot in road_labels:
        mx, my = transformer.transform(lon, lat)
        ax.text(mx, my, name, fontsize=7.5, fontweight="bold", color=WHITE,
                rotation=rot, ha="center", va="center", alpha=0.9, zorder=3,
                bbox=dict(boxstyle="round,pad=0.15", facecolor=DARK_TEXT, alpha=0.7, edgecolor="none"))

    # Neighbourhood labels (convert to Web Mercator)
    hoods = [
        (-79.383, 43.648, "Downtown"),
        (-79.550, 43.640, "Etobicoke"),
        (-79.250, 43.775, "Scarborough"),
        (-79.415, 43.785, "North York"),
    ]
    for lon, lat, name in hoods:
        mx, my = transformer.transform(lon, lat)
        ax.text(mx, my, name, fontsize=9, fontweight="bold", fontstyle="italic",
                color=DARK_TEXT, ha="center", va="center", alpha=0.9, zorder=3,
                bbox=dict(boxstyle="round,pad=0.2", facecolor=WHITE, alpha=0.6, edgecolor="none"))

    # Colorbar (use an opaque version for the legend)
    cbar_cmap = mcolors.LinearSegmentedColormap.from_list(
        "risk_cbar", [RISK_LOW, "#FFFF88", RISK_MED, RISK_HIGH], N=256
    )
    sm = plt.cm.ScalarMappable(cmap=cbar_cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.75, pad=0.02)
    cbar.set_label("Crash Risk (lambda)", fontsize=11, color=PRIMARY, fontweight="bold")
    cbar.set_ticks([0.15, 0.5, 0.85])
    cbar.set_ticklabels(["Low", "Medium", "High"])
    cbar.ax.tick_params(labelsize=10)

    # Hide axis ticks (the basemap handles geography)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])

    fig.savefig(OUT_DIR / "04_toronto_risk_heatmap.png", dpi=DPI,
                bbox_inches="tight", facecolor=WHITE)
    plt.close(fig)
    print("  Saved 04_toronto_risk_heatmap.png")


# ===================================================================
# Diagram 5: Dataset Pipeline & Feature Engineering Overview
# ===================================================================
def generate_pipeline_overview():
    fig, ax = plt.subplots(figsize=(16, 22))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 22)
    ax.axis("off")

    ax.text(8, 21.3, "Dataset Pipeline & Feature Engineering Overview",
            ha="center", va="center", fontsize=18, fontweight="bold", color=PRIMARY)

    # --- Row 1: Raw Data Sources ---
    y1 = 19.5
    ax.text(8, 20.3, "RAW DATA SOURCES", ha="center", fontsize=11,
            fontweight="bold", color=PRIMARY)
    _draw_box(ax, 2.5, y1, 3.8, 1.4,
              "Toronto Police\nCollisions\n(Excel, 618K records)",
              facecolor="#D6EAF8", edgecolor=PRIMARY, fontsize=8, fontweight="bold")
    _draw_box(ax, 8.0, y1, 3.8, 1.4,
              "Killed / Seriously\nInjured (KSI)\n(CSV, 18K records)",
              facecolor="#D6EAF8", edgecolor=PRIMARY, fontsize=8, fontweight="bold")
    _draw_box(ax, 13.5, y1, 3.8, 1.4,
              "Centreline Road\nNetwork\n(GeoJSON, 65K segments)",
              facecolor="#D6EAF8", edgecolor=PRIMARY, fontsize=8, fontweight="bold")

    # --- Row 1.5: Optional Enrichment ---
    y_enrich = 17.5
    ax.text(8, 18.35, "OPTIONAL ENRICHMENT", ha="center", fontsize=10,
            fontweight="bold", color=SECONDARY)
    enrich_items = [
        ("Weather\n(NOAA)", 2.0),
        ("TMC Traffic\nVolumes", 5.8),
        ("School\nLocations", 9.6),
        ("TTC GTFS\n(Transit)", 13.4),
    ]
    for label, x in enrich_items:
        _draw_box(ax, x, y_enrich, 2.8, 1.0, label,
                  facecolor="#D5F5E3", edgecolor=SECONDARY, fontsize=7.5)

    # Arrows from Row 1 down
    for x in [2.5, 8.0, 13.5]:
        _draw_arrow(ax, x, y1 - 0.7, x, y_enrich + 1.0, color=PRIMARY)
    # Enrichment arrows down
    for label, x in enrich_items:
        _draw_arrow(ax, x, y_enrich - 0.5, 8.0, 15.9, color=SECONDARY,
                    connectionstyle="arc3,rad=0.0")

    # --- Row 2: Spatial Join ---
    y2 = 15.3
    ax.text(8, 16.2, "DATA PROCESSING", ha="center", fontsize=11,
            fontweight="bold", color=PRIMARY)
    _draw_box(ax, 8.0, y2, 6.5, 1.3,
              "Spatial Join (BallTree indexing, UTM projection, 20m buffer)\n"
              "Crash-to-segment matching  |  Event-level + Aggregate joins",
              facecolor="#FCF3CF", edgecolor=ACCENT, fontsize=8, fontweight="bold")
    _draw_arrow(ax, 8.0, y_enrich - 0.5, 8.0, y2 + 0.65, color=PRIMARY)

    # --- Row 3: Feature Engineering ---
    y3 = 12.8
    ax.text(8, 14.0, "FEATURE ENGINEERING", ha="center", fontsize=11,
            fontweight="bold", color=PRIMARY)
    _draw_box(ax, 4.0, y3, 4.5, 1.4,
              "Panel Builder\n(configurable 1h–24h windows)\nTemporal train/val/test split",
              facecolor="#E8DAEF", edgecolor=PRIMARY, fontsize=8, fontweight="bold")

    _draw_box(ax, 12.0, y3, 5.5, 2.8,
              "7 Feature Groups\n"
              "─────────────────\n"
              "1. Weather (temp, precip, snow, wind)\n"
              "2. Road Geometry (length, class, degree)\n"
              "3. TMC Exposure (ped/cyclist/vehicle vol)\n"
              "4. School / Transit (zones, frequency)\n"
              "5. Lag Features (1d, 7d, 30d, rolling)\n"
              "6. Historical Profiles (crash/yr, ratios)\n"
              "7. Temporal Indicators (sin/cos encoding)",
              facecolor="#E8DAEF", edgecolor=PRIMARY, fontsize=7, fontweight="bold",
              textcolor=DARK_TEXT)

    _draw_arrow(ax, 8.0, y2 - 0.65, 4.0, y3 + 0.7, color=PRIMARY,
                connectionstyle="arc3,rad=0.1")
    _draw_arrow(ax, 8.0, y2 - 0.65, 12.0, y3 + 1.4, color=PRIMARY,
                connectionstyle="arc3,rad=-0.1")
    _draw_arrow(ax, 6.25, y3, 9.25, y3, color=SECONDARY)

    # --- Row 4: Model Training ---
    y4 = 9.5
    ax.text(8, 10.8, "MODEL TRAINING", ha="center", fontsize=11,
            fontweight="bold", color=PRIMARY)
    _draw_box(ax, 3.0, y4, 3.5, 1.3,
              "Temporal\nTrain / Val / Test\nSplit",
              facecolor="#FADBD8", edgecolor=RISK_HIGH, fontsize=8, fontweight="bold")
    _draw_box(ax, 8.0, y4, 3.5, 1.3,
              "Hurdle Model\n(2-stage)\nBinary + Poisson",
              facecolor="#FADBD8", edgecolor=RISK_HIGH, fontsize=8, fontweight="bold")
    _draw_box(ax, 13.0, y4, 3.5, 1.3,
              "Isotonic\nCalibration\n(Stage 1 probs)",
              facecolor="#FADBD8", edgecolor=RISK_HIGH, fontsize=8, fontweight="bold")

    _draw_arrow(ax, 4.0, y3 - 0.7, 3.0, y4 + 0.65, color=PRIMARY)
    _draw_arrow(ax, 4.75, y4, 6.25, y4, color=RISK_HIGH)
    _draw_arrow(ax, 9.75, y4, 11.25, y4, color=RISK_HIGH)

    # Tail weighting note
    ax.text(8.0, y4 - 1.0,
            "Tail-weighted sample weights:  w = 1 + alpha * log1p(y)  |  alpha=2.0, threshold=2",
            ha="center", fontsize=7, fontstyle="italic", color=SECONDARY)

    # --- Row 5: Inference & API ---
    y5 = 6.5
    ax.text(8, 7.8, "INFERENCE & SERVING", ha="center", fontsize=11,
            fontweight="bold", color=PRIMARY)

    _draw_box(ax, 3.0, y5, 3.0, 1.3,
              "Lambda Prediction\n"
              r"$\lambda = P_{cal} \times E[n|crash]$" + "\n"
              "clipped [0, 50]",
              facecolor="#FCF3CF", edgecolor=ACCENT, fontsize=7.5, fontweight="bold")
    _draw_box(ax, 8.0, y5, 3.0, 1.3,
              "Percentile\nThresholds\np70 / p90",
              facecolor="#FCF3CF", edgecolor=ACCENT, fontsize=8, fontweight="bold")
    _draw_box(ax, 13.0, y5, 3.0, 1.3,
              "Risk Labels\nLow / Medium / High",
              facecolor="#FCF3CF", edgecolor=ACCENT, fontsize=8, fontweight="bold")

    # Risk color dots
    for lbl, color, dx in [("Low", RISK_LOW, -0.6), ("Med", RISK_MED, 0.0), ("High", RISK_HIGH, 0.6)]:
        ax.plot(13.0 + dx, y5 - 0.45, "o", color=color, markersize=7, zorder=4)

    _draw_arrow(ax, 13.0, y4 - 0.65, 3.0, y5 + 0.65, color=ACCENT,
                connectionstyle="arc3,rad=0.2")
    _draw_arrow(ax, 4.5, y5, 6.5, y5, color=ACCENT)
    _draw_arrow(ax, 9.5, y5, 11.5, y5, color=ACCENT)

    # --- Row 6: API Endpoints ---
    y6 = 4.2
    ax.text(8, 5.3, "API & ROUTING", ha="center", fontsize=11,
            fontweight="bold", color=PRIMARY)

    _draw_box(ax, 3.5, y6, 4.5, 1.3,
              "/api/risk-predictions\nSegments in bounding box\n(max 500 segments)",
              facecolor="#D6EAF8", edgecolor=PRIMARY, fontsize=7.5, fontweight="bold")
    _draw_box(ax, 9.0, y6, 3.5, 1.3,
              "/api/risk-prediction\nSingle location\nrisk lookup",
              facecolor="#D6EAF8", edgecolor=PRIMARY, fontsize=7.5, fontweight="bold")
    _draw_box(ax, 13.5, y6, 3.5, 1.3,
              "/api/routes/\nsafety-aware\nDijkstra: time + risk",
              facecolor="#D6EAF8", edgecolor=PRIMARY, fontsize=7.5, fontweight="bold")

    _draw_arrow(ax, 8.0, y5 - 0.65, 3.5, y6 + 0.65, color=PRIMARY,
                connectionstyle="arc3,rad=0.15")
    _draw_arrow(ax, 13.0, y5 - 0.65, 9.0, y6 + 0.65, color=PRIMARY,
                connectionstyle="arc3,rad=0.0")
    _draw_arrow(ax, 13.0, y5 - 0.65, 13.5, y6 + 0.65, color=PRIMARY,
                connectionstyle="arc3,rad=-0.1")

    # Bottom note
    ax.text(8, 3.0,
            "iOS App  |  NetworkX road graph  |  Risk-weighted routing: cost = travel_time + beta * expected_crashes",
            ha="center", fontsize=8, color=SECONDARY, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", facecolor=LIGHT_GRAY, edgecolor=SECONDARY, alpha=0.5))

    fig.savefig(OUT_DIR / "05_data_pipeline_overview.png", dpi=DPI,
                bbox_inches="tight", facecolor=WHITE)
    plt.close(fig)
    print("  Saved 05_data_pipeline_overview.png")


# ===================================================================
# Diagram 6: Data Flow Diagram (DFD)
# ===================================================================
def generate_dataflow_diagram():
    fig, ax = plt.subplots(figsize=(22, 16))
    ax.set_xlim(0, 22)
    ax.set_ylim(0, 16)
    ax.axis("off")

    # Title
    ax.text(11, 15.5, "Data Flow Diagram — Toronto Crash Risk Prediction System",
            ha="center", va="center", fontsize=18, fontweight="bold", color=PRIMARY)

    # ── Helper: external entity (stadium-shape) ──
    def _draw_entity(ax, x, y, w, h, text, *, facecolor="#D6EAF8",
                     edgecolor=PRIMARY):
        _draw_box(ax, x, y, w, h, text, facecolor=facecolor, edgecolor=edgecolor,
                  fontsize=7, fontweight="bold", boxstyle="round,pad=0.25",
                  linewidth=2)

    # ── Helper: data store (open-ended rectangle) ──
    def _draw_store(ax, x, y, w, h, text):
        rect = mpatches.FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle="square,pad=0.0", facecolor="#FCF3CF", edgecolor=ACCENT,
            linewidth=1.5, zorder=2,
        )
        ax.add_patch(rect)
        # Top border line for data-store convention
        ax.plot([x - w / 2, x + w / 2], [y + h / 2, y + h / 2],
                color=ACCENT, linewidth=2.5, zorder=3)
        ax.text(x, y, text, ha="center", va="center", fontsize=6.5,
                fontweight="bold", color=DARK_TEXT, zorder=3)

    # ── Helper: process (rounded box with number) ──
    def _draw_process(ax, x, y, w, h, text, *, facecolor="#E8DAEF"):
        _draw_box(ax, x, y, w, h, text, facecolor=facecolor, edgecolor=PRIMARY,
                  fontsize=7, fontweight="bold", linewidth=2)

    # ── Helper: labeled arrow ──
    def _labeled_arrow(ax, x0, y0, x1, y1, label="", *,
                       color=PRIMARY, rad=0.0, fontsize=6):
        _draw_arrow(ax, x0, y0, x1, y1, color=color,
                    connectionstyle=f"arc3,rad={rad}")
        if label:
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            ax.text(mx, my + 0.18, label, ha="center", va="bottom",
                    fontsize=fontsize, fontstyle="italic", color=color, zorder=5,
                    bbox=dict(boxstyle="round,pad=0.1", facecolor=WHITE,
                              edgecolor="none", alpha=0.85))

    # ===============================================================
    # Column layout:  x=1.5  |  x=5.5  |  x=10  |  x=14.5  |  x=19
    # Row layout:     y=14   |  y=11   |  y=8   |  y=5     |  y=2
    # ===============================================================

    # ── External Entities (left column) ──
    _draw_entity(ax, 1.5, 13.5, 2.6, 1.0, "Toronto Police\nCollisions (Excel)")
    _draw_entity(ax, 1.5, 12.0, 2.6, 1.0, "KSI Dataset\n(CSV)")
    _draw_entity(ax, 1.5, 10.5, 2.6, 1.0, "Centreline Road\nNetwork (GeoJSON)")

    # Enrichment sources (top-right)
    _draw_entity(ax, 8.0, 14.5, 2.0, 0.7, "NOAA Weather",
                 facecolor="#D5F5E3", edgecolor=SECONDARY)
    _draw_entity(ax, 10.5, 14.5, 2.0, 0.7, "TMC Traffic",
                 facecolor="#D5F5E3", edgecolor=SECONDARY)
    _draw_entity(ax, 13.0, 14.5, 2.0, 0.7, "School Locations",
                 facecolor="#D5F5E3", edgecolor=SECONDARY)
    _draw_entity(ax, 15.5, 14.5, 2.0, 0.7, "TTC GTFS",
                 facecolor="#D5F5E3", edgecolor=SECONDARY)

    # ── Consumers (right column) ──
    _draw_entity(ax, 20.5, 6.5, 2.2, 1.0, "iOS App\nClient",
                 facecolor="#FADBD8", edgecolor=RISK_HIGH)
    _draw_entity(ax, 20.5, 4.5, 2.2, 1.0, "Developer\nDashboard",
                 facecolor="#FADBD8", edgecolor=RISK_HIGH)

    # ── Processes ──
    _draw_process(ax, 5.0, 12.0, 2.4, 1.6,
                  "1.0\nData Loader\n& Cleaning")
    _draw_process(ax, 9.0, 11.0, 2.8, 1.6,
                  "2.0\nSpatial Join\nBallTree · 20m buffer")
    _draw_process(ax, 9.0, 8.0, 2.8, 1.6,
                  "3.0\nPanel Builder\n& Feature Eng")
    _draw_process(ax, 14.0, 8.0, 2.8, 1.6,
                  "4.0\nHurdle Model\nTrain / Calibrate",
                  facecolor="#FADBD8")
    _draw_process(ax, 14.0, 5.5, 2.8, 1.4,
                  "5.0\nInference\nλ = Pcal × E[n|crash]",
                  facecolor="#FADBD8")
    _draw_process(ax, 9.0, 5.0, 2.8, 1.4,
                  "6.0\nRisk Labelling\np70 / p90 thresholds",
                  facecolor="#FADBD8")
    _draw_process(ax, 14.0, 2.5, 2.8, 1.4,
                  "7.0\nFlask API\nServing Layer")
    _draw_process(ax, 9.0, 2.5, 2.8, 1.2,
                  "8.0\nDijkstra Router\ntime + β·risk")

    # ── Data Stores ──
    _draw_store(ax, 5.0, 9.5, 2.6, 0.8, "D1: Raw Collision\nRecords")
    _draw_store(ax, 5.0, 7.5, 2.6, 0.8, "D2: Road Segment\nGeometry")
    _draw_store(ax, 5.0, 5.5, 2.6, 0.8, "D3: Enriched Panel\nDataset")
    _draw_store(ax, 18.0, 8.0, 2.6, 0.8, "D4: Trained Model\nArtifacts (.pkl)")
    _draw_store(ax, 18.0, 5.5, 2.6, 0.8, "D5: Lambda Risk\nMap (seg -> lambda)")
    _draw_store(ax, 5.0, 3.0, 2.6, 0.8, "D6: Road Graph\n(NetworkX)")

    # ===============================================================
    # Data Flows
    # ===============================================================

    # External → P1 (Data Loader)
    _labeled_arrow(ax, 2.8, 13.5, 3.8, 12.5, "collision records", rad=-0.1)
    _labeled_arrow(ax, 2.8, 12.0, 3.8, 12.0, "KSI records")
    _labeled_arrow(ax, 2.8, 10.5, 3.8, 11.5, "road segments", rad=0.1)

    # P1 → Data Stores
    _labeled_arrow(ax, 5.0, 11.2, 5.0, 9.9, "cleaned crashes")
    _labeled_arrow(ax, 5.0, 11.2, 5.0, 7.9, "geometry", rad=0.0)

    # D1, D2 → P2 (Spatial Join)
    _labeled_arrow(ax, 6.3, 9.5, 7.6, 11.2, "crash events", rad=0.15)
    _labeled_arrow(ax, 6.3, 7.8, 7.6, 10.8, "segments + IDs", rad=-0.1)

    # Enrichment → P2
    _labeled_arrow(ax, 8.0, 14.1, 8.5, 11.8, "weather", color=SECONDARY, rad=0.05)
    _labeled_arrow(ax, 10.5, 14.1, 9.5, 11.8, "volumes", color=SECONDARY, rad=0.05)
    _labeled_arrow(ax, 13.0, 14.1, 9.8, 11.8, "school flags", color=SECONDARY, rad=0.1)
    _labeled_arrow(ax, 15.5, 14.1, 10.2, 11.8, "transit freq", color=SECONDARY, rad=0.15)

    # P2 → P3
    _labeled_arrow(ax, 9.0, 10.2, 9.0, 8.8, "crash-segment\npairs + enrichment")

    # P3 → D3
    _labeled_arrow(ax, 7.6, 7.5, 6.3, 5.8, "feature matrix\n(40+ cols)", rad=0.1)

    # D3 → P4
    _labeled_arrow(ax, 6.3, 5.5, 12.6, 7.5, "train / val panels", rad=-0.1)

    # P4 → D4
    _labeled_arrow(ax, 15.4, 8.0, 16.7, 8.0, "model weights")

    # D4 → P5
    _labeled_arrow(ax, 18.0, 7.6, 15.4, 6.0, "trained models", rad=0.15)

    # D3 → P5
    _labeled_arrow(ax, 6.3, 5.2, 12.6, 5.5, "latest features", rad=0.05)

    # P5 → D5
    _labeled_arrow(ax, 15.4, 5.5, 16.7, 5.5, "λ per segment")

    # D5 → P6
    _labeled_arrow(ax, 16.7, 5.2, 10.4, 5.0, "lambda values", rad=0.1)

    # P6 → P7
    _labeled_arrow(ax, 10.4, 4.5, 12.6, 2.8, "risk labels", color=RISK_HIGH, rad=-0.1)

    # D2 → D6 (road geometry → graph)
    _labeled_arrow(ax, 5.0, 7.1, 5.0, 3.4, "road topology")

    # D5 → P8 (risk weights to router)
    _labeled_arrow(ax, 16.7, 5.2, 10.4, 3.0, "risk weights", rad=0.2)

    # D6 → P8
    _labeled_arrow(ax, 6.3, 2.8, 7.6, 2.5, "graph edges")

    # P8 → P7
    _labeled_arrow(ax, 10.4, 2.5, 12.6, 2.5, "route GeoJSON\n+ risk stats")

    # P7 → Consumers
    _labeled_arrow(ax, 15.4, 3.0, 19.4, 6.2, "segments + risk\nGeoJSON", color=RISK_HIGH, rad=-0.2)
    _labeled_arrow(ax, 15.4, 2.5, 19.4, 5.5, "route + stats", color=RISK_HIGH, rad=-0.1)

    # Consumers → P7 (requests)
    _labeled_arrow(ax, 19.4, 6.0, 15.4, 3.2,
                   "POST /risk-predictions\n/risk-prediction\n/routes/safety-aware",
                   color=PRIMARY, rad=-0.35)
    _labeled_arrow(ax, 19.4, 4.5, 15.4, 2.7,
                   "GET /health\n/model-features", color=PRIMARY, rad=-0.15)

    # ── Legend ──
    legend_y = 1.0
    legend_items = [
        ("External Entity", "#D6EAF8", PRIMARY),
        ("Process", "#E8DAEF", PRIMARY),
        ("Model Process", "#FADBD8", RISK_HIGH),
        ("Data Store", "#FCF3CF", ACCENT),
        ("Enrichment Source", "#D5F5E3", SECONDARY),
    ]
    for i, (label, fc, ec) in enumerate(legend_items):
        lx = 2.5 + i * 3.6
        box = FancyBboxPatch(
            (lx - 0.5, legend_y - 0.2), 1.0, 0.4,
            boxstyle="round,pad=0.1", facecolor=fc, edgecolor=ec,
            linewidth=1.5, zorder=2,
        )
        ax.add_patch(box)
        ax.text(lx + 0.8, legend_y, label, fontsize=7, fontweight="bold",
                color=DARK_TEXT, va="center", zorder=3)

    fig.savefig(OUT_DIR / "06_dataflow_diagram.png", dpi=DPI,
                bbox_inches="tight", facecolor=WHITE)
    plt.close(fig)
    print("  Saved 06_dataflow_diagram.png")


# ===================================================================
# Diagram 7: SHAP Feature Importance Summary
# ===================================================================
def generate_shap_summary():
    # Synthetic SHAP values mirroring the 7 feature groups
    features = [
        "hist_crash_per_yr",
        "rolling_mean_7d",
        "crashes_1d_ago",
        "tmc_vehicle_vol",
        "segment_length",
        "tmc_ped_vol",
        "precipitation",
        "temperature",
        "transit_freq",
        "snow_depth_mm",
        "is_school_zone",
        "hour_sin",
        "is_weekend",
        "month_sin",
        "road_degree",
    ]
    # Mean |SHAP| values (synthetic, realistic ordering)
    shap_vals = np.array([
        0.142, 0.118, 0.095, 0.082, 0.071, 0.063, 0.054, 0.048,
        0.039, 0.035, 0.028, 0.022, 0.018, 0.013, 0.010,
    ])

    # Feature group color mapping
    group_colors = {
        "hist_crash_per_yr": PRIMARY, "rolling_mean_7d": PRIMARY,
        "crashes_1d_ago": PRIMARY,
        "tmc_vehicle_vol": ACCENT, "tmc_ped_vol": ACCENT,
        "segment_length": SECONDARY, "road_degree": SECONDARY,
        "precipitation": "#5B9BD5", "temperature": "#5B9BD5",
        "snow_depth_mm": "#5B9BD5",
        "transit_freq": "#8E6BAD", "is_school_zone": "#8E6BAD",
        "hour_sin": "#888888", "is_weekend": "#888888",
        "month_sin": "#888888",
    }

    # Sort by importance (ascending for horizontal barh)
    order = np.argsort(shap_vals)
    features_sorted = [features[i] for i in order]
    shap_sorted = shap_vals[order]
    colors_sorted = [group_colors[f] for f in features_sorted]

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_title("Feature Importance — Mean |SHAP| Values",
                 fontsize=16, fontweight="bold", color=PRIMARY, pad=20)

    bars = ax.barh(range(len(features_sorted)), shap_sorted,
                   color=colors_sorted, edgecolor=WHITE, linewidth=0.5, height=0.7)

    ax.set_yticks(range(len(features_sorted)))
    ax.set_yticklabels(features_sorted, fontsize=10)
    ax.set_xlabel("Mean |SHAP value|", fontsize=12, fontweight="bold", color=PRIMARY)
    ax.set_xlim(0, max(shap_vals) * 1.15)

    # Value labels on bars
    for i, (val, bar) in enumerate(zip(shap_sorted, bars)):
        ax.text(val + 0.002, i, f"{val:.3f}", va="center", fontsize=9, color=DARK_TEXT)

    # Group legend
    from matplotlib.patches import Patch
    legend_items = [
        Patch(facecolor=PRIMARY, label="Lag / Historical"),
        Patch(facecolor=ACCENT, label="TMC Exposure"),
        Patch(facecolor=SECONDARY, label="Road Geometry"),
        Patch(facecolor="#5B9BD5", label="Weather"),
        Patch(facecolor="#8E6BAD", label="School / Transit"),
        Patch(facecolor="#888888", label="Temporal"),
    ]
    ax.legend(handles=legend_items, loc="lower right", fontsize=9, framealpha=0.9,
              title="Feature Group", title_fontsize=10)

    # Annotation
    ax.text(0.98, 0.02,
            "Historical crash profile and recent lag features\n"
            "dominate predictions — supports temporal modeling choice",
            transform=ax.transAxes, fontsize=8, fontstyle="italic", color=SECONDARY,
            ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.4", facecolor=LIGHT_GRAY,
                      edgecolor=SECONDARY, alpha=0.7))

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.savefig(OUT_DIR / "07_shap_summary.png", dpi=DPI,
                bbox_inches="tight", facecolor=WHITE)
    plt.close(fig)
    print("  Saved 07_shap_summary.png")


# ===================================================================
# Diagram 8: Calibration Curve (Reliability Diagram)
# ===================================================================
def generate_calibration_curve():
    rng = np.random.default_rng(42)

    # Synthetic calibration data — 10 bins
    bin_edges = np.linspace(0, 1, 11)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Before calibration: overconfident at high end, underconfident at low end
    fraction_pos_before = np.array([
        0.005, 0.025, 0.065, 0.115, 0.175, 0.280, 0.400, 0.580, 0.740, 0.880
    ])
    # After isotonic calibration: much closer to diagonal
    fraction_pos_after = np.array([
        0.008, 0.048, 0.092, 0.148, 0.210, 0.305, 0.410, 0.510, 0.610, 0.720
    ])
    # Mean predicted probability in each bin
    mean_predicted_before = np.array([
        0.01, 0.05, 0.10, 0.15, 0.22, 0.32, 0.45, 0.55, 0.68, 0.82
    ])
    mean_predicted_after = np.array([
        0.01, 0.05, 0.10, 0.15, 0.21, 0.30, 0.40, 0.50, 0.60, 0.72
    ])
    # Sample counts per bin (heavily skewed — most samples are low-risk)
    bin_counts = np.array([
        42000, 18000, 8500, 4200, 2100, 1100, 600, 350, 180, 90
    ])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), height_ratios=[3, 1],
                                    gridspec_kw={"hspace": 0.08})

    fig.suptitle("Calibration Curve — Stage 1 Binary Classifier",
                 fontsize=16, fontweight="bold", color=PRIMARY, y=0.95)

    # ── Top: Reliability diagram ──
    # Perfect calibration line
    ax1.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Perfect calibration")

    # Before calibration
    ax1.plot(mean_predicted_before, fraction_pos_before, "s-",
             color=ACCENT, linewidth=2, markersize=8, label="Before calibration",
             markeredgecolor=WHITE, markeredgewidth=1)

    # After isotonic calibration
    ax1.plot(mean_predicted_after, fraction_pos_after, "o-",
             color=PRIMARY, linewidth=2.5, markersize=8, label="After isotonic calibration",
             markeredgecolor=WHITE, markeredgewidth=1)

    # Shade improvement region
    for i in range(len(bin_centers)):
        if abs(mean_predicted_before[i] - fraction_pos_before[i]) > \
           abs(mean_predicted_after[i] - fraction_pos_after[i]):
            ax1.plot([mean_predicted_after[i]], [fraction_pos_after[i]], "o",
                     color=RISK_LOW, markersize=4, zorder=5)

    ax1.set_ylabel("Fraction of Positives (observed)", fontsize=12,
                   fontweight="bold", color=PRIMARY)
    ax1.set_xlim(-0.02, 1.02)
    ax1.set_ylim(-0.02, 1.02)
    ax1.legend(fontsize=10, loc="upper left", framealpha=0.9)
    ax1.set_xticklabels([])
    ax1.grid(True, alpha=0.3)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Brier score annotation
    brier_before = np.mean((mean_predicted_before - fraction_pos_before) ** 2)
    brier_after = np.mean((mean_predicted_after - fraction_pos_after) ** 2)
    ax1.text(0.98, 0.35,
             f"Brier Score\n"
             f"Before:  {brier_before:.4f}\n"
             f"After:    {brier_after:.4f}\n"
             f"Improvement: {(1 - brier_after / brier_before) * 100:.0f}%",
             transform=ax1.transAxes, fontsize=10, fontfamily="monospace",
             ha="right", va="top", color=DARK_TEXT,
             bbox=dict(boxstyle="round,pad=0.5", facecolor=LIGHT_GRAY,
                       edgecolor=PRIMARY, alpha=0.9))

    # ── Bottom: Histogram of predictions ──
    ax2.bar(bin_centers, bin_counts, width=0.08, color=PRIMARY, alpha=0.6,
            edgecolor=WHITE, label="Sample count per bin")
    ax2.set_xlabel("Mean Predicted Probability", fontsize=12,
                   fontweight="bold", color=PRIMARY)
    ax2.set_ylabel("Count", fontsize=11, fontweight="bold", color=PRIMARY)
    ax2.set_xlim(-0.02, 1.02)
    ax2.set_yscale("log")
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.text(0.5, 0.85, "Most samples concentrated in low-probability bins (class imbalance)",
             transform=ax2.transAxes, fontsize=8, fontstyle="italic",
             color=SECONDARY, ha="center", va="top")

    fig.savefig(OUT_DIR / "08_calibration_curve.png", dpi=DPI,
                bbox_inches="tight", facecolor=WHITE)
    plt.close(fig)
    print("  Saved 08_calibration_curve.png")


# ===================================================================
# Diagram 9: System Architecture (Deployment View)
# ===================================================================
def generate_system_architecture():
    fig, ax = plt.subplots(figsize=(20, 12))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 12)
    ax.axis("off")

    ax.text(10, 11.5, "System Architecture — Deployment View",
            ha="center", va="center", fontsize=18, fontweight="bold", color=PRIMARY)

    # ── iOS Client Zone ──
    client_rect = mpatches.FancyBboxPatch(
        (0.5, 5.5), 4.0, 5.0, boxstyle="round,pad=0.3",
        facecolor="#F0F8FF", edgecolor=PRIMARY, linewidth=2, linestyle="--",
        alpha=0.5, zorder=0)
    ax.add_patch(client_rect)
    ax.text(2.5, 10.2, "CLIENT", ha="center", fontsize=10,
            fontweight="bold", color=PRIMARY, fontstyle="italic")

    _draw_box(ax, 2.5, 9.0, 3.2, 1.2, "iOS App (SwiftUI)\nMapKit + Risk Overlay\nRoute Comparison UI",
              facecolor="#FADBD8", edgecolor=RISK_HIGH, fontsize=8, fontweight="bold")
    _draw_box(ax, 2.5, 7.2, 3.2, 1.0, "HTTP Client\nJSON Request/Response\nLocation Services",
              facecolor="#FADBD8", edgecolor=RISK_HIGH, fontsize=7.5)
    _draw_arrow(ax, 2.5, 8.4, 2.5, 7.7, color=RISK_HIGH)

    # ── Server Zone ──
    server_rect = mpatches.FancyBboxPatch(
        (5.5, 0.8), 9.5, 10.0, boxstyle="round,pad=0.3",
        facecolor="#F5F5F5", edgecolor=SECONDARY, linewidth=2, linestyle="--",
        alpha=0.5, zorder=0)
    ax.add_patch(server_rect)
    ax.text(10.25, 10.5, "SERVER (Flask Application)", ha="center", fontsize=10,
            fontweight="bold", color=SECONDARY, fontstyle="italic")

    # API Layer
    _draw_box(ax, 7.5, 9.0, 3.5, 1.4,
              "Flask API + CORS\n"
              "Endpoints:\n"
              "/risk-predictions\n"
              "/risk-prediction\n"
              "/routes/safety-aware",
              facecolor="#D6EAF8", edgecolor=PRIMARY, fontsize=7, fontweight="bold")

    # Model Runtime
    _draw_box(ax, 12.5, 9.0, 3.5, 1.4,
              "Model Runtime\n"
              "TemporalCountModelTrainer\n"
              "Inference + Calibration\n"
              "Lambda Computation",
              facecolor="#FADBD8", edgecolor=RISK_HIGH, fontsize=7, fontweight="bold")

    _draw_arrow(ax, 9.25, 9.0, 10.75, 9.0, color=PRIMARY)
    ax.text(10.0, 9.2, "predict()", fontsize=7, fontstyle="italic",
            color=PRIMARY, ha="center")

    # Routing Engine
    _draw_box(ax, 7.5, 6.5, 3.5, 1.4,
              "Routing Engine\n"
              "NetworkX DiGraph\n"
              "Dijkstra: time + beta*risk\n"
              "snap_to_graph / path_edges",
              facecolor="#E8DAEF", edgecolor=PRIMARY, fontsize=7, fontweight="bold")

    _draw_arrow(ax, 7.5, 8.3, 7.5, 7.2, color=PRIMARY)

    # Spatial Index
    _draw_box(ax, 12.5, 6.5, 3.5, 1.4,
              "Spatial Index\n"
              "Shapely R-tree\n"
              "Bounding-box queries\n"
              "Segment lookup (max 500)",
              facecolor="#E8DAEF", edgecolor=PRIMARY, fontsize=7, fontweight="bold")

    _draw_arrow(ax, 12.5, 8.3, 12.5, 7.2, color=PRIMARY)

    # Feature Pipeline
    _draw_box(ax, 7.5, 4.0, 3.5, 1.4,
              "Feature Pipeline\n"
              "PanelBuilder\n"
              "7 feature groups\n"
              "StandardScaler",
              facecolor="#D5F5E3", edgecolor=SECONDARY, fontsize=7, fontweight="bold")

    _draw_arrow(ax, 7.5, 5.8, 7.5, 4.7, color=SECONDARY)

    # Data Loader
    _draw_box(ax, 12.5, 4.0, 3.5, 1.4,
              "Data Loader\n"
              "spatial_join_fast\n"
              "BallTree (UTM, 20m)\n"
              "Centreline ID matching",
              facecolor="#D5F5E3", edgecolor=SECONDARY, fontsize=7, fontweight="bold")

    _draw_arrow(ax, 9.25, 4.0, 10.75, 4.0, color=SECONDARY)
    _draw_arrow(ax, 12.5, 5.8, 12.5, 4.7, color=SECONDARY)

    # Config
    _draw_box(ax, 10.0, 1.8, 3.0, 1.0,
              "Config\nCOLLISION_COLUMNS\nKSI_COLUMNS\nPanelConfig",
              facecolor="#FCF3CF", edgecolor=ACCENT, fontsize=7)
    _draw_arrow(ax, 10.0, 2.3, 7.5, 3.3, color=ACCENT,
                connectionstyle="arc3,rad=0.1")
    _draw_arrow(ax, 10.0, 2.3, 12.5, 3.3, color=ACCENT,
                connectionstyle="arc3,rad=-0.1")

    # ── Storage Zone ──
    storage_rect = mpatches.FancyBboxPatch(
        (15.8, 0.8), 3.8, 10.0, boxstyle="round,pad=0.3",
        facecolor="#FFFAF0", edgecolor=ACCENT, linewidth=2, linestyle="--",
        alpha=0.5, zorder=0)
    ax.add_patch(storage_rect)
    ax.text(17.7, 10.5, "STORAGE", ha="center", fontsize=10,
            fontweight="bold", color=ACCENT, fontstyle="italic")

    _draw_box(ax, 17.7, 9.0, 3.0, 1.0,
              "Model Artifact\ntoronto_temporal_count\n_model.pkl",
              facecolor="#FCF3CF", edgecolor=ACCENT, fontsize=7, fontweight="bold")
    _draw_arrow(ax, 14.25, 9.2, 16.2, 9.2, color=ACCENT)
    ax.text(15.2, 9.4, "load_model()", fontsize=6.5, fontstyle="italic", color=ACCENT)

    _draw_box(ax, 17.7, 7.0, 3.0, 1.0,
              "Road Network\nCentreline GeoJSON\n65K segments",
              facecolor="#FCF3CF", edgecolor=ACCENT, fontsize=7, fontweight="bold")
    _draw_arrow(ax, 14.25, 6.5, 16.2, 6.8, color=ACCENT)

    _draw_box(ax, 17.7, 5.0, 3.0, 1.0,
              "Collision Data\nExcel (618K)\nCSV (18K KSI)",
              facecolor="#FCF3CF", edgecolor=ACCENT, fontsize=7, fontweight="bold")
    _draw_arrow(ax, 14.25, 4.2, 16.2, 4.8, color=ACCENT)

    _draw_box(ax, 17.7, 3.0, 3.0, 1.0,
              "Weather Cache\nhistoricalweather.csv\nNOAA lookups",
              facecolor="#FCF3CF", edgecolor=ACCENT, fontsize=7, fontweight="bold")

    _draw_box(ax, 17.7, 1.5, 3.0, 0.8,
              "Lambda Risk Map\n(in-memory cache)",
              facecolor="#FCF3CF", edgecolor=ACCENT, fontsize=7)

    # ── Network arrows (Client ↔ Server) ──
    _draw_arrow(ax, 4.5, 7.5, 5.75, 8.5, color=PRIMARY,
                connectionstyle="arc3,rad=-0.15")
    ax.text(5.0, 8.4, "HTTPS\nJSON", fontsize=7, fontweight="bold",
            color=PRIMARY, ha="center", rotation=30,
            bbox=dict(boxstyle="round,pad=0.15", facecolor=WHITE,
                      edgecolor=PRIMARY, alpha=0.8))
    _draw_arrow(ax, 5.75, 8.0, 4.5, 7.0, color=RISK_HIGH,
                connectionstyle="arc3,rad=-0.15")
    ax.text(5.0, 7.0, "GeoJSON\nresponse", fontsize=6.5, fontstyle="italic",
            color=RISK_HIGH, ha="center", rotation=30)

    # ── Legend ──
    legend_items = [
        ("Client Layer", "#FADBD8", RISK_HIGH),
        ("API / Routing", "#D6EAF8", PRIMARY),
        ("Processing", "#D5F5E3", SECONDARY),
        ("Storage", "#FCF3CF", ACCENT),
    ]
    for i, (label, fc, ec) in enumerate(legend_items):
        lx = 1.5 + i * 3.2
        box = FancyBboxPatch(
            (lx - 0.4, 0.3), 0.8, 0.35,
            boxstyle="round,pad=0.08", facecolor=fc, edgecolor=ec,
            linewidth=1.5, zorder=2)
        ax.add_patch(box)
        ax.text(lx + 0.7, 0.48, label, fontsize=7.5, fontweight="bold",
                color=DARK_TEXT, va="center", zorder=3)

    fig.savefig(OUT_DIR / "09_system_architecture.png", dpi=DPI,
                bbox_inches="tight", facecolor=WHITE)
    plt.close(fig)
    print("  Saved 09_system_architecture.png")


# ===================================================================
# Diagram 10a: Crash Count Distribution (Histogram)
# ===================================================================
def _make_crash_counts():
    """Shared synthetic crash count data for 10a–10c."""
    rng = np.random.default_rng(42)
    counts = np.concatenate([
        np.zeros(88000),
        np.ones(8000),
        np.full(2500, 2),
        np.full(800, 3),
        np.full(350, 4),
        np.full(150, 5),
        np.full(80, 6),
        np.full(50, 7),
        rng.integers(8, 15, size=70),
    ])
    return counts


def generate_crash_count_histogram():
    counts = _make_crash_counts()
    total = len(counts)

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.suptitle("Crash Count Distribution — Zero-Inflated Target",
                 fontsize=16, fontweight="bold", color=PRIMARY, y=0.97)

    bins = np.arange(0, 16) - 0.5
    n_vals, _, patches = ax.hist(counts, bins=bins, color=PRIMARY, edgecolor=WHITE,
                                  linewidth=0.8, alpha=0.85)

    # Color the zero bar differently
    patches[0].set_facecolor(SECONDARY)

    ax.set_yscale("log")
    ax.set_xlabel("Crash Count per Window", fontsize=12, fontweight="bold", color=PRIMARY)
    ax.set_ylabel("Number of Windows (log scale)", fontsize=12, fontweight="bold", color=PRIMARY)
    ax.set_xticks(range(0, 15))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.2, axis="y")

    # Annotate zero percentage
    zero_pct = (counts == 0).sum() / total * 100
    ax.annotate(f"{zero_pct:.0f}% zero\n(no crash)",
                xy=(0, (counts == 0).sum()), xytext=(3, 60000),
                fontsize=11, fontweight="bold", color=SECONDARY,
                arrowprops=dict(arrowstyle="->", color=SECONDARY, lw=2),
                bbox=dict(boxstyle="round,pad=0.4", facecolor=WHITE,
                          edgecolor=SECONDARY, alpha=0.9))

    # Annotate tail
    tail_count = (counts >= 3).sum()
    tail_pct = tail_count / total * 100
    ax.annotate(f"{tail_pct:.1f}% tail (3+)\n{tail_count:,} windows",
                xy=(5, (counts == 5).sum()), xytext=(8, 5000),
                fontsize=10, fontweight="bold", color=RISK_HIGH,
                arrowprops=dict(arrowstyle="->", color=RISK_HIGH, lw=2),
                bbox=dict(boxstyle="round,pad=0.4", facecolor=WHITE,
                          edgecolor=RISK_HIGH, alpha=0.9))

    # Bottom note
    ax.text(0.98, 0.02,
            "Zero-inflated distribution motivates the two-stage Hurdle architecture:\n"
            "Stage 1 filters zeros, Stage 2 models positive counts",
            transform=ax.transAxes, fontsize=9, fontstyle="italic", color=SECONDARY,
            ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.4", facecolor=LIGHT_GRAY,
                      edgecolor=SECONDARY, alpha=0.7))

    fig.savefig(OUT_DIR / "10a_crash_count_distribution.png", dpi=DPI,
                bbox_inches="tight", facecolor=WHITE)
    plt.close(fig)
    print("  Saved 10a_crash_count_distribution.png")


# ===================================================================
# Diagram 10b: Risk Label Split (Pie Chart)
# ===================================================================
def generate_risk_label_split():
    fig, ax = plt.subplots(figsize=(9, 7))
    fig.suptitle("Risk Label Distribution — Percentile Thresholds",
                 fontsize=16, fontweight="bold", color=PRIMARY, y=0.97)

    # Based on p70/p90 thresholds: 70% Low, 20% Medium, 10% High
    label_counts = [70000, 20000, 10000]
    label_names = ["Low\n(70%)", "Medium\n(20%)", "High\n(10%)"]
    label_colors = [RISK_LOW, RISK_MED, RISK_HIGH]

    wedges, texts, autotexts = ax.pie(
        label_counts, labels=label_names, colors=label_colors,
        autopct=lambda p: f"{int(p * sum(label_counts) / 100):,}",
        startangle=90, pctdistance=0.65,
        wedgeprops=dict(edgecolor=WHITE, linewidth=2.5),
        textprops=dict(fontsize=12, fontweight="bold"),
    )
    for at in autotexts:
        at.set_fontsize(10)
        at.set_color(WHITE)
        at.set_fontweight("bold")

    # Bottom note
    ax.text(0.5, -0.08,
            "Thresholds: p70 separates Low/Medium, p90 separates Medium/High\n"
            "Applied to predicted lambda values across all road segments",
            transform=ax.transAxes, fontsize=9, fontstyle="italic", color=SECONDARY,
            ha="center", va="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor=LIGHT_GRAY,
                      edgecolor=SECONDARY, alpha=0.7))

    fig.savefig(OUT_DIR / "10b_risk_label_split.png", dpi=DPI,
                bbox_inches="tight", facecolor=WHITE)
    plt.close(fig)
    print("  Saved 10b_risk_label_split.png")


# ===================================================================
# Diagram 10c: Hurdle Model Filtering (Stacked Bar)
# ===================================================================
def generate_hurdle_filtering():
    fig, ax = plt.subplots(figsize=(9, 7))
    fig.suptitle("Hurdle Model — Stage Filtering Breakdown",
                 fontsize=16, fontweight="bold", color=PRIMARY, y=0.97)

    categories = ["All\nWindows", "Stage 1\nOutput (P>0.5)", "Stage 2\nInput"]
    zero_counts = [88000, 0, 0]
    pos_counts = [12000, 12000, 12000]
    remaining = [0, 88000, 0]

    bar_width = 0.55
    x_pos = np.arange(len(categories))

    ax.bar(x_pos, pos_counts, bar_width, label="Crash > 0 (positive)",
           color=RISK_HIGH, edgecolor=WHITE, linewidth=1)
    ax.bar(x_pos, zero_counts, bar_width, bottom=pos_counts,
           label="Crash = 0 (zero)", color=SECONDARY, edgecolor=WHITE, linewidth=1)
    ax.bar(x_pos, remaining, bar_width, bottom=[p + z for p, z in zip(pos_counts, zero_counts)],
           label="Filtered out", color=LIGHT_GRAY, edgecolor=WHITE, linewidth=1)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(categories, fontsize=10, fontweight="bold")
    ax.set_ylabel("Window Count", fontsize=12, fontweight="bold", color=PRIMARY)
    ax.legend(fontsize=9, loc="upper right", framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.2, axis="y")

    # Annotate the filtering
    ax.annotate("88% filtered\nby Stage 1",
                xy=(1, 12000), xytext=(1.6, 60000),
                fontsize=10, fontweight="bold", color=PRIMARY,
                arrowprops=dict(arrowstyle="->", color=PRIMARY, lw=2))

    # Bottom note
    ax.text(0.5, -0.08,
            "Tail-weighted sample weights: w = 1 + 2.0 * log1p(y) for y >= 2\n"
            "Upweights rare high-count windows to improve tail accuracy",
            transform=ax.transAxes, fontsize=9, fontstyle="italic", color=SECONDARY,
            ha="center", va="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor=LIGHT_GRAY,
                      edgecolor=SECONDARY, alpha=0.7))

    fig.savefig(OUT_DIR / "10c_hurdle_filtering.png", dpi=DPI,
                bbox_inches="tight", facecolor=WHITE)
    plt.close(fig)
    print("  Saved 10c_hurdle_filtering.png")


# ===================================================================
# Main
# ===================================================================
def main():
    print(f"\nGenerating board diagrams → {OUT_DIR}/\n")
    generate_hurdle_architecture()
    generate_confusion_matrix()
    generate_correlation_heatmap()
    generate_toronto_risk_heatmap()
    generate_pipeline_overview()
    generate_dataflow_diagram()
    generate_shap_summary()
    generate_calibration_curve()
    generate_system_architecture()
    generate_crash_count_histogram()
    generate_risk_label_split()
    generate_hurdle_filtering()
    print(f"\nDone — 12 diagrams saved at {DPI} DPI.\n")


if __name__ == "__main__":
    main()
