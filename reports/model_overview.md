# Toronto Road Crash Risk Prediction Model — Overview

## What is this project?

We built a machine learning model that predicts the likelihood of a traffic crash happening on any road segment in Toronto, for any given hour. The city's road network has over 65,000 individual segments, and the model assigns a risk score to each one based on the characteristics of the road, the time of day, the weather, how much traffic the road carries, and its crash history.

The model powers a backend API that supports two main use cases:

1. **Risk mapping** — showing which road segments are highest-risk at any given time
2. **Safety-aware routing** — finding a route between two points that balances travel time against crash risk (similar to how Google Maps finds the "fastest" route, but with a "safest" option)

---

## What does the model predict?

The model predicts **expected crash count per road segment per hour** — a number called lambda (λ). For most segments at most times, this number is very small (e.g., 0.003 expected crashes per hour). For a busy arterial during rush hour in winter rain, it might be higher (e.g., 0.05).

This lambda value is then converted into:

- A **probability of at least one crash** in the next hour: P = 1 − e^(−λ)
- A **risk score for routing**: safer routes avoid segments with high λ values

The model does **not** predict whether a specific crash will happen — crashes are rare, random events. Instead, it estimates the underlying risk level, which changes depending on conditions.

---

## What makes it predictive?

The model uses 62 input features organized into six categories. Each category contributes a different type of signal:

### Road characteristics (what kind of road is this?)
- **Road type**: local street, collector, minor arterial, major arterial, expressway, etc.
- **Physical properties**: segment length, whether it's one-way, how many roads meet at each end (intersection complexity)

### Traffic volume (how busy is this road?)
- **Daily volume**: average vehicles per day (ranges from 3 to 65,000 across Toronto)
- **Speed**: average speed and 85th/95th percentile speeds
- **Peak-hour volume**: morning and evening rush hour traffic counts
- **Exposure**: total vehicle-kilometres per year — the standard traffic safety metric for how "used" a road is

### Weather (what are conditions like today?)
- **Temperature**: daily average in Celsius (below freezing = icy road risk)
- **Precipitation**: rainfall in mm
- **Snow depth**: accumulated snow on the ground in mm
- **Binary flags**: is it freezing? is there any precipitation today?

### Time of day (when is it?)
- **Hour of day**: encoded as a cyclical pattern so the model understands 11pm is close to midnight
- **Day of week**: weekday vs weekend patterns
- **Month and season**: winter months have different crash patterns than summer

### Historical crash profile (how dangerous has this road been?)
- **Annual crash rate**: average crashes per year on this segment
- **Time-of-day pattern**: what fraction of this segment's crashes historically happen during this part of the day
- **Weekend pattern**: what fraction happen on weekends

### Recent crash history
- **Lag features**: whether crashes occurred on this segment in the past 1 hour, 24 hours, or 7 days
- **Rolling averages**: recent crash trends

---

## How was it trained?

The model was trained on **11 years of Toronto crash data** (2014–2025):

- **618,254** general traffic collision records
- **18,957** killed-or-seriously-injured (KSI) records
- **65,133** road network segments
- **11,562** segments with traffic volume data
- **4,070** days of weather data (daily, expanded to hourly)

The data was split chronologically — the model trains on earlier years and is tested on the most recent data, simulating real-world deployment where you always predict the future based on the past.

The model itself is a **Gradient Boosted Decision Tree** with a Poisson loss function, which is the standard approach for predicting event counts (like crashes) that are rare and non-negative.

---

## What changed in this update?

The previous version of the model had critical flaws that made it appear to work but not actually predict anything meaningful. This update fixed those issues:

| Before | After |
|--------|-------|
| Model memorized specific road segment IDs instead of learning patterns | Model learns from road characteristics that generalize to any segment |
| Road type feature was broken (string converted to zeros) | Road type properly encoded as distinct categories |
| No weather data | Temperature, precipitation, snow, and freezing conditions included |
| No traffic volume data | Daily traffic counts, speed, peak-hour volumes, and exposure included |
| Time-of-day features had no variation | Cyclical time encoding captures rush hour, nighttime, weekend patterns |
| 17 mostly broken features | 62 working features across 6 meaningful categories |

---

## Key Questions and Answers

### Why a Poisson model instead of Negative Binomial?

**Short answer:** Poisson is the right starting point and is likely sufficient for our use case.

| | Poisson | Negative Binomial |
|--|---------|-------------------|
| **Assumption** | Mean = Variance | Variance > Mean (overdispersion) |
| **When to use** | Counts that are rare and roughly random | Counts with more variance than a Poisson would predict |
| **Our situation** | Crashes per segment per hour are extremely rare (~99.4% of hours have zero crashes). The mean is very close to zero, so overdispersion is hard to detect at this granularity. |
| **Practical difference** | Simpler, faster, well-supported in gradient boosting frameworks | Would require custom implementation or a different modeling framework (e.g., statsmodels GLM) |
| **Recommendation** | **Use Poisson for now.** If we move to daily or weekly granularity where segments regularly have 1–5+ crashes per window, overdispersion becomes measurable and Negative Binomial would be worth testing. At hourly granularity, the data is too sparse for the distinction to matter. |

### Why hourly granularity? What about daily or weekly?

Each granularity has trade-offs:

| Granularity | Pros | Cons |
|-------------|------|------|
| **Hourly** | Captures rush-hour vs nighttime patterns. Most useful for real-time routing ("is this road dangerous right now?"). Aligns with weather changes. | Crashes are extremely rare per segment per hour (~0.1% of rows have a crash). Lag features are mostly zeros. Needs large training set. |
| **Daily** | Crashes are still sparse but less extreme (~1-3% non-zero). Weather data naturally fits. Lag features more useful. | Loses hour-of-day signal within the window. Less useful for real-time routing. |
| **Weekly** | Most segments have non-zero crash counts. Lag features carry real signal. Smaller dataset, faster training. | Loses all intra-week timing. Can't distinguish Monday morning rush from Sunday midnight. Not useful for real-time applications. |

**Current choice: hourly.** This was chosen because the downstream use case (safety-aware routing) benefits from knowing that a road is more dangerous at 8am than at 3am. The sparsity problem is addressed through historical crash profile features (annual rate, hour-bucket ratios) which aggregate years of history into informative per-segment summaries.

**If the team decides real-time routing is not the priority**, switching to daily windows would make the model stronger statistically while still capturing day-to-day weather variation.

### How accurate is the model?

The model achieved on the held-out test set (most recent data, never seen during training):

- **MAE = 0.059** — on average, predictions are off by 0.059 crashes per segment per hour
- **RMSE = 1.659** — root mean squared error, penalizing large errors more heavily
- **Poisson Deviance = 0.141** — measures how well the predicted distribution matches reality

These numbers are hard to interpret in isolation because the target is extremely sparse (99.4% zeros). The important question is whether the model **ranks** segments correctly — do the segments it calls high-risk actually have more crashes? This should be validated with a rank-ordering analysis or a calibration plot.

### What about the 82% of segments with no traffic volume data?

Only 17.8% of Toronto's road segments (11,562 out of 65,133) have traffic volume measurements. For the remaining segments, all traffic features are zero.

The model handles this by learning that "zero traffic data" is a distinct signal — these tend to be smaller local streets where formal traffic counts haven't been conducted. The road type features (local, collector, arterial) partially compensate, since road class is a rough proxy for traffic volume.

**Improving coverage** of traffic data would meaningfully improve predictions for those segments.

### What about the weather being daily, not hourly?

The weather data comes from NOAA daily summaries. Every hour on the same day gets the same temperature and precipitation value. This means the model knows "today is a rainy winter day" but not "it's raining right now at 3pm."

**Impact:** The model can learn seasonal and day-level weather patterns (freezing days are more dangerous) but cannot react to within-day weather changes (sudden afternoon thunderstorm).

**Improvement path:** Environment Canada publishes actual hourly weather observations for Toronto Pearson airport. Switching to that data source would give the model true hour-by-hour weather variation — the single biggest remaining improvement for prediction quality.

### Could this model be used for other cities?

The model architecture is city-agnostic. The same approach (road network + crash data + weather + traffic volumes → Poisson GBDT) would work for any city with similar open data. The features would need to be rebuilt from that city's data, and the model retrained, but no code changes would be required beyond data loading.

### What data does the model need to run in production?

For the API to serve live predictions:

1. **Road network geometry** — Toronto Centreline (already loaded)
2. **Trained model file** — `toronto_temporal_count_model.pkl` (updated with each retrain)
3. **Panel snapshot** — `panel_latest.parquet` (pre-computed features for all segments at the latest time window)
4. **Current time** — to select the right temporal features (hour, day, season)
5. **(Optional) Live weather** — if connected to a weather API, predictions update with real-time conditions

---

## Files and Components

| Component | File | Purpose |
|-----------|------|---------|
| Training script | `train_temporal_model.py` | Runs the full pipeline: load data → build features → train model → save outputs |
| Data loader | `src/data_processing/data_loader.py` | Reads crash data, road network, weather, traffic volumes |
| Feature builder | `src/feature_engineering/panel_builder.py` | Constructs the feature panel (road + time + weather + history per segment per hour) |
| Model trainer | `src/models/model_trainer.py` | Trains the Poisson GBDT model, evaluates, calibrates, saves |
| Backend API | `backend-api/app.py` | Serves predictions and safety-aware routing over HTTP |
| iOS app | `ios-app/` | Mobile client for viewing risk maps and getting routed directions |
| Trained model | `outputs/models/toronto_temporal_count_model.pkl` | Serialized trained model (62 features) |
| Inference panel | `outputs/reports/panel_latest.parquet` | Pre-computed features for all 65,133 segments |
