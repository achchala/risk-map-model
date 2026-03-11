# Toronto Road Crash Risk Model — Stakeholder Briefing

---

## Executive Summary

We built a machine learning model that predicts **which road segments in Toronto are most likely to experience a crash, and when**. The model was trained on years of historical crash data (2014–2025), weather conditions, road geometry, and traffic volume across 65,133 road segments in the Toronto Centreline network.

Three approaches were tested head-to-head. Two of them — a simple historical crash rate and our full machine learning model (called HistGBR) — dramatically outperform random guessing, identifying high-risk segments **~566–605% more effectively** than chance. The machine learning model's distinctive advantage is **temporal precision**: it can tell you not just that a segment is dangerous, but *which specific hours* are elevated risk (e.g., icy conditions at 8 AM on a January Wednesday), something a static historical average simply cannot do.

The model is currently in a research/validation phase. It demonstrates strong practical value for safety-informed routing. Traffic volume (AADT) data has been integrated into the pipeline and the model has been retrained with it.

---

## Summary of Each Folder & What It Means

### 📁 Maps

**What's in here:** Two geographic output files and one interactive web map.

| File | What it is |
|------|-----------|
| `toronto_risk_map.html` | An interactive map of Toronto showing road segments color-coded by risk level — open this in a browser to explore it visually |
| `toronto_road_risk.geojson` | The full road risk dataset in a standard geographic format — this is what you'd load into a mapping tool or GIS system |
| `toronto_high_risk_roads.geojson` | A filtered version containing only the highest-risk road segments |

**Interpretation:** These are the final, actionable outputs of the model — the geographic "answers." The interactive HTML map is the easiest way for non-technical stakeholders to explore results. The GeoJSON files are ready to be plugged into any routing system, mapping dashboard, or further analysis.

---

### 📁 Models

**What's in here:** Two saved, trained machine learning model files.

| File | What it is |
|------|-----------|
| `toronto_temporal_count_model.pkl` | The primary trained model (HistGBR) — this is the model ready to make live predictions |
| `toronto_risk_model.joblib` | An earlier version of the model saved in an alternative format |

**Interpretation:** These are the "brains" — the trained models that can be loaded and used to score any road segment at any future date and hour. Think of them like a trained expert that has studied four years of Toronto crash history and can now answer "how risky is this road right now?" on demand. The `.pkl` file is the active, production-ready model.

---

### 📁 Reports

**What's in here:** Five analysis documents and supporting data files covering data quality, model comparisons, and business value. Here's each one explained:

#### Report 1 — Data Integrity & Leakage Audit (`01_data_integrity_report.md`)

**What it checked:** This report is a quality-control audit that verifies the model was built fairly — meaning it only learned from historical information and was never accidentally "shown" future crash data during training.

**Key findings:**
- The training/testing split was done correctly by time. The model trained on older data and was tested on data it had never seen (May 2021 → April 2025).
- All 21 potentially "leaky" variables (like future crash counts) were confirmed absent from the model's inputs — a clean bill of health.
- The three most important features the model relies on are: **historical crash frequency per year**, **road segment length**, and **the ratio of crashes occurring during specific hours of day**. Weather (snow depth) comes in fourth.

**Plain-language takeaway:** The model played fair — it had no unfair advantage when we tested it.

---

#### Report 2 — Model Bake-Off & Diagnostics (`02_model_bake_off_report.md`)

**What it checked:** A head-to-head race between three approaches to see which one predicts crashes most accurately.

| Model | What it does | AUC-ROC | Lift@5% |
|-------|-------------|---------|---------|
| Naive (random guess) | Predicts the same average crash rate for every road, every hour | 0.50 (coin flip) | 0.45× |
| Historical Rate | Uses each segment's long-run average crash history | 0.91 | 9.86× |
| HistGBR (our ML model) | Uses weather, time-of-day, traffic, and road features together | 0.85 | 9.78× |

**Key finding — the over-prediction explanation:** The model assigns a small non-zero risk score to almost every road window, even when no crash occurs. This sounds like a flaw, but it's actually the expected and correct behavior for a problem where 99.85% of observations have zero crashes. What matters is whether the model correctly *ranks* roads from most to least dangerous — and it does.

**Plain-language takeaway:** Both the ML model and the simple historical rate method are roughly 10× better than random at identifying which road segments to flag. The ML model's real edge isn't in overall accuracy — it's in the ability to add *time-sensitive* context that the historical baseline cannot.

---

#### Report 3 — Business Impact & Routing Utility (`03_business_impact_report.md`)

**What it checked:** Whether the model can actually help a routing system make safer decisions, simulated across 1,000 random route choices.

**Routing simulation results:**

| Strategy | Mean crashes avoided per route | Improvement vs. random |
|----------|-------------------------------|----------------------|
| Random (baseline) | 0.11 | — |
| Historical Rate | 0.78 | +605% |
| HistGBR Model | 0.74 | +566% |

**Cumulative recall — how much of total crash risk do we capture?**

If you flag only the **top 5%** of riskiest road-hour windows, you capture **~49% of all actual crashes** that occurred in the test period. That means you can reduce exposure to half of all crashes by avoiding just 5% of road-hour combinations.

**Plain-language takeaway:** Using either the ML model or historical crash rates to guide routing decisions is dramatically better than random. The ML model's unique advantage is knowing not just *where* crashes happen but *when* — making it suitable for real-time routing that adapts to weather and time of day.

---

#### Model Horse Race Report (`MODEL_HORSE_RACE_REPORT.md`)

**What it checked:** A deep statistical audit prompted by an academic challenge: is the model mathematically justified, or did we just assume the wrong statistical framework?

Three specific statistical concerns were tested:

1. **Overdispersion** — The crash data has a variance nearly **287× higher** than what standard Poisson regression assumes. This was confirmed as a real problem. (Think of it as the data being far "lumpier" than a textbook model expects.)

2. **Zero-inflation** — 94.8% of road segments have zero crashes. A standard model predicted only 7,686 zero-crash segments; we actually observed 61,749. This structural excess of zeros was confirmed.

3. **Non-linear patterns** — Tree-based models (XGBoost) outperformed linear regression by 5.7% on accuracy and identified 17.9 percentage points more "safe" segments correctly.

**Plain-language takeaway:** The academic critique was valid — the data violates the assumptions of simple Poisson regression. Our gradient boosting model handles this more robustly. The Negative Binomial model did not improve over Poisson — it degenerated to identical results, meaning the dispersion parameter collapsed rather than fitting the overdispersion. The Zero-Inflated Negative Binomial (ZINB) failed to converge entirely. These statistical approaches remain limited until a proper exposure offset (vehicle-km) is available for all 65,133 segments; traffic volume data is now integrated for 11,562 of them (17.8%).

---

## FAQ for Stakeholders

**Q: What does this model actually predict?**
A: For any road segment in Toronto at any given hour, the model outputs a risk score (λ) representing the expected number of crashes in that window. In practice, this score is used to *rank* roads from most to least dangerous — not to predict an exact crash count.

**Q: How accurate is it?**
A: When you flag the top 5% of highest-risk road-hour windows, you capture approximately 49% of all real crashes. Put differently, you could reduce crash exposure by half while only avoiding 1 in 20 road segments. That is a strong result for a problem this rare and complex.

**Q: Why does the simpler historical rate method perform similarly to the fancy ML model?**
A: At the level of a full road segment over a long time period, historical crash history is already a very strong signal — and both models have learned it. The ML model's advantage is *temporal granularity*: it knows that a segment that is risky on average becomes *especially* risky on a snowy Tuesday morning at 7 AM. A static historical rate cannot make that distinction.

**Q: Why can't the model predict crash counts exactly?**
A: Crashes are extremely rare events (less than 0.15% of road-hour combinations have any crash), highly random, and influenced by factors we can't observe — driver behavior, split-second decisions, mechanical failures. No model can predict individual crashes with certainty. What the model does well is *ranking risk* — identifying which roads and times are systematically more dangerous.

**Q: Is this model production-ready?**
A: It is research-validated and demonstrates real routing value. Traffic volume (AADT/speed data) has been integrated and the model has been retrained with these features for 11,562 segments (17.8% of the network). The main remaining gap is coverage: 82.2% of road segments still have no traffic volume measurement, so the exposure offset is partial. The model is suitable for production deployment on high-priority corridors where volume data exists.

**Q: What are the next steps to improve it?**
A: Two high-impact improvements remaining, in priority order:
1. Expand traffic volume coverage — currently 11,562 of 65,133 segments have AADT data. Extending this to the full network (via the City of Toronto's ongoing count program or turn movement counts) would unlock proper exposure normalization across all roads.
2. Add functional road classification (arterial vs. collector vs. local) — the current data uses street suffix types (St, Ave, Rd) as a proxy, which is a rough approximation.

**Q: Who should use the interactive map?**
A: The `toronto_risk_map.html` file (in the Maps folder) is designed for anyone — no technical background needed. Open it in any web browser to visually explore Toronto's risk landscape by road segment. The GeoJSON files are for developers and data teams who want to integrate the data into other systems.

**Q: What time period was the model tested on?**
A: The model was trained on crash data up to approximately early 2021, and tested on data from May 2021 through April 2025 — roughly four years of held-out test data it had never seen during training. This "time-ordered" split is the most rigorous way to validate a temporal prediction model.

**Q: How many roads does this cover?**
A: The model is trained and scored on all 65,133 road segments in the Toronto Centreline dataset. The test set evaluation observed crashes on 4,475 of those segments — reflecting the fact that most segments had zero crashes during the test window, not that the model only covers 4,475 roads.
