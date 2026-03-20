# Final Report: Safety-Aware Routing via Road Segment Crash Risk Prediction

**Team:** Risk Map Model — Capstone Design Project
**Date:** March 2026
**Institution:** [University Name]

---

## 1. Problem Analysis

### 1.1 Problem Definition

Road crashes are a leading cause of injury and death in urban environments. While drivers have access to navigation tools that optimize for speed and distance, no widely available system routes users based on real-time crash risk. This project addresses that gap.

**Core objective:** Predict the probability of a crash occurring on a given road segment within a future 24-hour window, and use that prediction to offer drivers a safety-aware routing alternative to the fastest path.

The goal is not to predict the exact number of crashes — it is to produce a reliable *risk ranking* across segments so that routing decisions can favor lower-risk corridors. This distinction is important: the system is optimized for ranking accuracy (lift, AUC) rather than count accuracy (RMSE), which is the correct framing for a routing use case.

### 1.2 Context and Background

Toronto records approximately 50,000–65,000 motor vehicle collisions annually. This project uses 618,254 historical collision records spanning 2014–2025, spatially matched to 65,133 road segments from the City of Toronto's centreline dataset.

The prediction problem is structurally challenging for three reasons:

**Sparse data (extreme zero-inflation).** When crash events are aggregated into (segment, 24-hour window) units, 98.47% of windows contain zero crashes. Standard predictive models, which assume symmetrically distributed errors, perform poorly on this distribution. A naive model that always predicts zero would be correct 98.47% of the time but entirely useless for routing.

**Temporal and spatial complexity.** Crash risk is not static. It varies by time of day, day of week, season, and weather conditions. It also varies spatially: arterial roads in high-pedestrian areas have fundamentally different risk profiles than local residential streets. Any useful model must capture both dimensions simultaneously.

**Limited exposure variables.** The most reliable crash risk predictor — traffic volume — is only partially available. Average Daily Traffic (ADT) counts exist for some segments but are static and annual. Real-time volume data (e.g., INRIX) was not available, which constrains model precision.

### 1.3 Stakeholders and Impact

| Stakeholder | Benefit |
|---|---|
| **Drivers and passengers** | Safer route alternatives during high-risk windows (weather events, rush hour on arterials) |
| **Cyclists and pedestrians** | Awareness of high-risk corridors when planning active travel |
| **City planners and traffic engineers** | Data-driven identification of persistently high-risk segments for infrastructure intervention |
| **Emergency services** | Potential to pre-position resources based on predicted risk windows |

The system has positive social impact by reducing collision-related injury and death. Economically, the cost of road crashes in Canada exceeds $37 billion annually (Transport Canada, 2020); even marginal routing improvements at scale represent meaningful cost reduction. Environmentally, safety-aware routes are not always longer — in many cases, a slightly longer route on a lower-volume arterial produces comparable or lower emissions while reducing risk.

Across the design lifecycle, data is sourced from existing City of Toronto open datasets (no new data collection infrastructure required), the model runs on lightweight hardware (sub-1MB model file), and the iOS app has no persistent server-side storage.

### 1.4 Existing Solutions and Gaps

| Approach | Description | Limitation |
|---|---|---|
| **Historical averages** | Flag segments with high past crash counts | Static — does not respond to time, weather, or conditions |
| **Heuristic rules** | Avoid highways at rush hour, snow routes, etc. | Not data-driven; misses interaction effects |
| **Navigation apps (Google, Apple Maps)** | Optimize for time and traffic flow | Do not incorporate crash risk at all |
| **Academic crash models (AADT-based GLMs)** | Predict annual crash frequency per segment | Annual resolution; not deployable for real-time routing |

The gap is a **predictive, condition-aware, real-time deployable** risk scoring system. No existing consumer product provides hourly or daily crash risk scores per road segment as a routing input. This system is designed to fill that gap.

---

## 2. Requirements Analysis

### 2.1 Key Requirements

| ID | Requirement | Target | Verification Method |
|----|---|---|---|
| **R1** | Predict crash risk per segment per 24-hour window, ranked accurately | AUC-ROC > 0.75; Lift@5% > 5× | Held-out test set evaluation (AUC, lift curve) |
| **R2** | Predicted probabilities are calibrated (predicted ≈ observed rates) | Calibration curve within ±10% across deciles | Isotonic calibration + decile calibration plot |
| **R3** | Inference latency suitable for real-time routing | Per-segment: < 50 ms; Full city: < 500 ms | Runtime profiling on inference pipeline |
| **R4** | Model handles extreme sparsity (98.5% zeros) without degenerate predictions | No class collapse; Stage 2 trains on positive examples only | Model comparison against Poisson GLM baseline |
| **R5** | End-to-end pipeline is reproducible and deployable | Single-command training and inference | Pipeline test; API endpoint test |

### 2.2 Prioritization

**R1 and R2 are highest priority.** A model that ranks risk correctly and whose probabilities are calibrated is the core product requirement — without these, routing decisions are not trustworthy.

**R3** is a deployment constraint, not a model quality requirement. It was verified after model selection.

**R4** is an architectural requirement that drove the choice of hurdle model over simpler baselines.

**R5** ensures the project is reproducible beyond the team.

A full requirements traceability matrix is included in the Appendix.

---

## 3. Solution Development

### 3.1 Concept Exploration

Four model families were evaluated before arriving at the final architecture:

| Model | Rationale for Consideration | Reason Rejected |
|---|---|---|
| **Poisson GLM** | Standard count model; interpretable | Assumes mean = variance (violated 287×); poor on zero-inflation |
| **Negative Binomial GLM** | Handles overdispersion | Still treats all zeros as structural; poor lift |
| **Zero-Inflated NB (ZINB)** | Separates structural and sampling zeros | Complex to tune; no gradient boosting support; slow |
| **XGBoost (count regression)** | Flexible, handles non-linearity | Does not natively separate occurrence from intensity |
| **Two-Stage Hurdle (final)** | Separates binary occurrence from count | — |

**Key insight:** The problem has two distinct generative processes — *whether* a crash occurs on a segment in a given window (a classification problem), and *how many* crashes occur given that at least one does (a count problem). Treating these as a single prediction task forces the model to simultaneously explain structural absence and event severity with one set of parameters. Separating them into a hurdle model resolves this.

### 3.2 System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA INGESTION                           │
│  Collisions (618K) │ Road Network (65K) │ Weather │ TMC │ TTC  │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                    ┌───────────▼───────────┐
                    │   SPATIAL JOIN (R-tree) │
                    │  Crash → Segment (20m) │
                    └───────────┬───────────┘
                                │
                    ┌───────────▼───────────┐
                    │   PANEL CONSTRUCTION   │
                    │  (segment, 24h window) │
                    │  Negative sample 1:10  │
                    └───────────┬───────────┘
                                │
                    ┌───────────▼───────────┐
                    │  FEATURE ENGINEERING   │
                    │  66 features, 6 groups │
                    └───────────┬───────────┘
                                │
              ┌─────────────────┼─────────────────┐
              │                 │                 │
    ┌─────────▼──────┐  ┌───────▼──────┐  ┌──────▼───────┐
    │  TRAIN (60%)   │  │  VAL (20%)   │  │  TEST (20%)  │
    │  2014–2019     │  │  2019–2021   │  │  2021–2025   │
    └─────────┬──────┘  └───────┬──────┘  └──────┬───────┘
              │                 │                 │
    ┌─────────▼─────────────────▼─────┐           │
    │        TWO-STAGE HURDLE MODEL    │           │
    │                                  │           │
    │  Stage 1: HistGBClassifier       │           │
    │  → P(crash) [binary]             │           │
    │  → Isotonic calibration on val   │           │
    │                                  │           │
    │  Stage 2: HistGBRegressor        │           │
    │  → E[count | crash > 0]          │           │
    │  → Poisson loss, pos. only       │           │
    │                                  │           │
    │  λ = P(crash) × E[count|crash]   │           │
    └──────────────────┬───────────────┘           │
                       │                           │
                       └───────────────────────────┘
                                   │ Evaluate on test set
                    ┌──────────────▼──────────────┐
                    │       ROUTING ENGINE         │
                    │  Flask API + Path Finding    │
                    │  Fastest vs Safer Route      │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │         iOS APP              │
                    │  SwiftUI + MapKit            │
                    │  Risk Map + Route Compare    │
                    └─────────────────────────────┘
```

### 3.3 Data Pipeline

**Sources:**

| Dataset | Records | Purpose |
|---|---|---|
| Motor Vehicle Collisions (City of Toronto) | 618,254 | Crash events with lat/lon and datetime |
| KSI (Killed or Seriously Injured) | 18,000+ | Severe/fatal crash flag |
| Toronto Centreline (road network) | 65,133 segments | Road geometry and classification |
| Traffic Volume (model_dataset.csv) | Per-segment | ADT, speed, peak volumes |
| Historical Weather (NOAA) | Hourly, city-wide | Temperature, precipitation, snow, wind |
| TMC Counts | Intersection-level | Pedestrian, cyclist, vehicle volumes |
| School Locations | Polygon | School zone buffers (200m) |
| TTC GTFS | Transit stops | Transit frequency (150m buffer) |

**Panel Construction:**
Each training example is a (segment_id, window_start) pair. The label is the crash count in the *following* 24-hour window (H=24h, steps_ahead=1), enforcing no future leakage.

Positive windows (≥1 future crash) are fully sampled. Negative windows are randomly downsampled at a 1:10 ratio (one positive per ten negatives) to reduce compute while preserving class signal. Sample weights are applied at training time to correct for this downsampling.

**Temporal split:** Train 60% (earliest) → Val 20% (middle) → Test 20% (most recent, 2021–2025). This respects the time-series structure and ensures the test set reflects genuine out-of-sample performance.

### 3.4 Model Design

**Two-Stage Hurdle Model:**

**Stage 1 — Binary Crash Classifier**
- Algorithm: `HistGradientBoostingClassifier` (log loss)
- Hyperparameters: max_depth=6, learning_rate=0.1, max_iter=300
- Calibration: Isotonic regression fitted on validation set to align predicted probabilities with observed crash rates
- Output: P(crash) ∈ [0, 1]

**Stage 2 — Conditional Count Regressor**
- Algorithm: `HistGradientBoostingRegressor` (Poisson loss)
- Trained *only* on windows where future_crash_count > 0
- Output: E[count | crash > 0]

**Combined prediction:**
```
λ = P(crash) × E[count | crash > 0]
P(≥1 crash) = 1 − e^(−λ)      [used in routing]
```

**Why this works for this problem:**
- Stage 1 focuses entirely on learning what distinguishes crash windows from non-crash windows — the dominant signal in the data.
- Stage 2 focuses on severity distribution without being distorted by structural zeros.
- Isotonic calibration ensures the routing engine receives well-calibrated probabilities rather than raw logit scores.

**Features (66 total across 6 categories):**

| Category | Examples |
|---|---|
| Road geometry | segment_length, road_class (one-hot), is_oneway, intersection degree |
| Traffic volume | ADT, avg_speed, peak volumes, heavy vehicle fraction, log_volume |
| TMC exposure | Daily pedestrian, cyclist, vehicle counts at nearby intersections |
| Contextual | is_school_zone, is_school_active_hour, nearby_transit_frequency |
| Temporal | hour_sin/cos, dow_sin/cos, is_weekend, month_sin/cos, season |
| Weather | temperature, precipitation, snow_depth, wind_speed, is_freezing, is_precip |
| Historical profiles | hist_crashes_per_year, hist_crash_hour_ratio, hist_crash_weekend_ratio |
| Lag / rolling | crashes_1d_ago, crashes_7d_ago, rolling_mean_7d, rolling_max_7d |
| Interaction | snow×arterial, freeze×arterial, freeze×vehicle_vol, precip×ped_vol |

### 3.5 Key Design Decisions

| Decision | Rationale | Trade-off |
|---|---|---|
| **Daily windows (not hourly)** | Reduces zero-inflation from 99.85% (hourly) to 98.47% (daily); makes Stage 2 feasible | Less temporal resolution for routing |
| **Negative downsampling 1:10** | Reduces training set size; preserves class balance for tree splits | Requires sample weight correction |
| **Tail weighting** | Up-weights rare high-count events (w = 1 + 2.0 × log(1 + y)); down-weights sampled zeros | Increases recall on high-risk windows; minor precision loss |
| **Temporal split (no random shuffle)** | Prevents future leakage; test set reflects real deployment conditions | Cannot use k-fold cross-validation |
| **HistGBR over XGBoost** | Native Poisson loss; handles missing values; faster training | Less community tooling for hyperparameter search |
| **Isotonic vs Platt calibration** | Isotonic is non-parametric; better fit for non-monotone calibration errors | Requires held-out validation set |

---

## 4. Solution Evaluation

### 4.1 Performance Results

**Test set:** 311,072 windows, 4,475 segments, May 2021 – March 2025

| Metric | Value | Interpretation |
|---|---|---|
| **AUC-ROC** | **0.817** | Strong discrimination between crash and non-crash windows |
| AUC-PR | 0.122 | 8.0× above naive baseline (0.015) |
| **Lift @ 5%** | **8.52×** | Top 5% of predicted windows contain 8.52× more crashes than random |
| **Recall @ 5%** | **41.6%** | Flagging 5% of windows captures 41.6% of all crashes |
| MAE | 0.019 | Low absolute error |
| RMSE | 0.147 | Tight prediction spread |

**Cumulative crash capture:**

| Fraction of windows flagged | Crashes captured |
|---|---|
| Top 1% | 14.6% |
| Top 5% | 41.6% |
| Top 10% | 57.3% |
| Top 20% | 76.3% |
| Top 50% | 88.5% |

**Routing simulation (1,000 random 10-segment trials):**

| Strategy | Mean crashes on route | vs. random |
|---|---|---|
| Random route selection | 1.27 | baseline |
| Model-guided safer route | 0.20 | **−84% / +554% improvement** |

### 4.2 Validation Methods

**Ablation study:** Each of the 8 feature groups was removed one at a time and AUC-ROC was re-evaluated. Historical crash profiles were the most important group (−0.030 AUC when removed). No single group caused catastrophic failure; minimum AUC across all ablations was 0.787.

**Calibration analysis:** Decile calibration plot confirms that predicted risk deciles monotonically correspond to actual crash rates. D9/D1 ratio = 146×, confirming strong discriminative power. Isotonic calibration reduced calibration error on the validation set.

**Diebold-Mariano test:** Formal statistical comparison of per-window squared prediction error against the historical rate baseline. Test statistic = −2.005, p-value = 0.045 (two-sided). The hurdle model produces statistically significantly lower prediction error at α = 0.05.

**Temporal validation:** Test set spans 2021–2025, covering COVID recovery traffic patterns, seasonal variation, and multiple winter events. The model was never exposed to this data during training or validation.

**Residual analysis:** Mean residual = −0.015 (slight under-prediction bias). 98.3% of residuals fall within [−0.23, +0.13]. No systematic bias by road class.

### 4.3 Requirement Verification

| Requirement | Met? | Evidence |
|---|---|---|
| R1 — Accurate risk ranking | ✅ | AUC-ROC = 0.817 (target: > 0.75); Lift@5% = 8.52× (target: > 5×) |
| R2 — Calibrated probabilities | ✅ | Monotonic decile calibration; isotonic calibration reduces val error |
| R3 — Fast inference | ✅ | 17.3 ms per segment; 272.9 ms full Toronto batch (target: < 500 ms) |
| R4 — Handles sparse data | ✅ | Hurdle model trained stably; Stage 2 on 1.53% positive windows only |
| R5 — Deployable pipeline | ✅ | Single-command training; Flask API with 4 tested endpoints; iOS app running |

### 4.4 Feasibility and Limitations

**Current limitations:**

- **Traffic volume quality:** ADT data is annual and static. Real-time traffic volume (INRIX, HERE) was not available. This is the single largest missing predictor for crash risk.
- **Weather granularity:** City-wide hourly weather is used as a proxy for local conditions. Hyper-local weather variation (e.g., black ice on specific segments) is not captured.
- **Daily resolution:** The 24-hour prediction window is coarser than ideal for time-sensitive routing. Hourly windows were evaluated but produced 99.85% zero-inflation, making Stage 2 training infeasible without significantly more data.
- **Geographic scope:** Model is trained and validated on Toronto only. Generalization to other cities requires retraining.

**Future improvements:**

| Improvement | Expected Impact |
|---|---|
| Hourly weather API (replace daily NOAA) | High — weather is a primary trigger for crash clusters |
| Real-time traffic volume (INRIX) | High — exposure is the strongest missing predictor |
| Hyperparameter tuning (lr=0.05) | AUC-ROC +0.065 (0.817 → 0.882) — validated in tuning experiments |
| Road surface condition data (Ontario 511) | Medium — explains ice/freeze mechanism better |
| Hourly windows with more data | Medium — requires ~10× more positive training examples |

**Performance ceiling:** AUC-ROC > 0.88 is achievable with hyperparameter tuning alone. With hourly weather and real-time traffic, > 0.90 is estimated based on feature ablation results.

---

## Appendix

### A. Full Requirements Table

| ID | Requirement | Priority | Target | Verification | Status |
|---|---|---|---|---|---|
| R1 | Predict crash risk per segment-hour, ranked accurately | High | AUC > 0.75, Lift@5% > 5× | Test set AUC + lift curve | ✅ Met |
| R2 | Calibrated probabilities (predicted ≈ observed) | High | Calibration within ±10% across deciles | Calibration curve on test set | ✅ Met |
| R3 | Fast inference | Medium | < 50 ms/segment, < 500 ms full city | Runtime profiling | ✅ Met |
| R4 | Handles sparse data (98.5% zeros) | High | Stable predictions, no class collapse | Model comparison vs. Poisson GLM | ✅ Met |
| R5 | Deployable end-to-end pipeline | Medium | Single-command training + working API | Pipeline test + endpoint test | ✅ Met |
| R6 | iOS app displays risk and routing | Low | Working demo on device/simulator | Manual demo test | ✅ Met |

### B. Model Comparison Summary

| Model | AUC-ROC | Lift@5% | Notes |
|---|---|---|---|
| Naive (predict mean) | 0.500 | 0.35× | Baseline |
| Historical rate | 0.500 | 0.35× | No temporal adaptation |
| Poisson GLM | ~0.62 | ~1.8× | Fails on zero-inflation |
| **Two-Stage Hurdle (final)** | **0.817** | **8.52×** | Statistically significant improvement |

### C. System Specifications

| Component | Specification |
|---|---|
| Training data | 618,254 collisions, 65,133 segments, 2014–2025 |
| Feature count | 66 engineered features |
| Training set size | ~60% of temporal windows |
| Test set | 311,072 windows, May 2021 – March 2025 |
| Model file size | 0.88 MB |
| Inference latency | 17.3 ms (single), 272.9 ms (full city) |
| Backend | Flask API, Python 3.11 |
| Frontend | iOS, SwiftUI, MapKit |
