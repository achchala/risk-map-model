# Tail-weighting implementation Q&A (answers from current codebase)

---

## A) Training dataset distribution and sampling

**How is the weekly training panel constructed?**  
- **Case-control sampled.** It is **not** the full population of segment-weeks.
- **Positives:** All `(segment_id, window_start)` pairs with **current-window** `crash_count > 0` (from sparse aggregation of event-level crashes).
- **Negatives:** A **sampled** set of `(segment_id, window_start)` pairs with **0** crashes in the current window. Sampling is from the Cartesian space of **active segments** (segments that appear in event-level data) × **all weekly windows** in the event date range. Any pair that is already a positive is excluded.

**Negative sampling rate:**  
- **Default: 10× negatives per positive.**  
- Set by `negative_multiplier=10` in `build_weekly_sampled_future_panel()` (`panel_builder.py`). The training script does **not** pass this argument, so 10 is used.  
- `target_neg = negative_multiplier * n_pos`; then up to `target_neg` negative pairs are drawn (with replacement over segment and window, then de-duplicated and excluding positives).

**Stratification:**  
- **No.** Sampling is **uniform** over `(segment_id, window_start)`: `rng.choice(active_segments, ...)` and `rng.choice(all_windows, ...)`. No stratification by road class or time.

**Sample weights:**  
- **Yes.** Weights are computed and applied.
- **Formula:**  
  - `approx_zero_pairs = max(active_segments × all_windows − n_pos, 1)` (approximate count of zero-label segment-weeks in the full universe).  
  - `n_zero_sampled` = number of rows in the panel with `future_crash_count == 0`.  
  - For rows with `future_crash_count == 0`: `sample_weight = w0 = approx_zero_pairs / n_zero_sampled`.  
  - For rows with `future_crash_count > 0`: `sample_weight = 1.0`.  
- **Where applied:** In `TemporalCountModelTrainer.train_temporal_count_model()`, when `sample_weight_col="sample_weight"` is passed (as in `train_temporal_model.py`), `sample_weight_train = train_data[sample_weight_col].astype(float).values` is passed to `model.fit(X_train_scaled, y_train, sample_weight=sample_weight_train)`. So the **training** objective is reweighted; evaluation (MAE, RMSE, etc.) and diagnostics are **unweighted**.

---

## B) Exact target and units

**Target column:**  
- **`future_crash_count`** (integer count in the **next** window).  
- Set explicitly in `train_temporal_model.py`: `target_col="future_crash_count"` in `train_temporal_count_model(...)`.

**Window/horizon config:**  
- **`window_size_hours = 168`**, **`horizon_hours = 168`** (1 week each).  
- Defined in `train_temporal_model.py`: `PanelConfig(window_size_hours=24 * 7, horizon_hours=24 * 7)`.  
- Future label is built by join: `future_window_start = window_start + 168h`, then left-join of crash counts on `(segment_id, future_window_start)`; missing → 0.

**Model output units:**  
- **Predicted expected count per segment-week (μ),** i.e. per 168h window.  
- **Not** per-hour or per-traversal. Callers (e.g. routing) convert to per-hour or to traversal probability as needed (e.g. `λ_hour = λ_window / 168`, then `1 - exp(-λ_hour * t)` for travel time `t`).

---

## C) Model, loss, and training code path

**Model class and init:**  
- **`sklearn.ensemble.HistGradientBoostingRegressor`** with:
  - `loss="poisson"`
  - `max_depth=6`
  - `learning_rate=0.1`
  - `max_iter=300`
  - `random_state=self.random_state` (trainer default 42)  
- **Not** set in current code: `min_samples_leaf`, `l2_regularization`, `early_stopping`.

**Does `.fit()` pass `sample_weight`?**  
- **Yes.** `self.model.fit(X_train_scaled, y_train, sample_weight=sample_weight_train)` when `sample_weight_col` is provided and present in `train_data`.  
- **Shape/alignment:** `sample_weight_train` is `train_data[sample_weight_col].astype(float).values`, so 1D, length = `len(train_data)`, same row order as `X_train` / `y_train` (both come from the same `train_data` after temporal split). No extra filtering is applied between building the panel and the split, so rows align.

**Capping: training vs inference:**  
- **Training:** The model is fit on **uncapped** targets and predictions; **no** λ cap is applied during `fit()` or to `y_train`.  
- **Evaluation:** After `model.predict(X_test_scaled)`, predictions are clipped: `y_pred = np.clip(y_pred, 0.0, None)` and then `y_pred = np.clip(y_pred, 0.0, self.lambda_cap)` if `lambda_cap` is set. So **capping is evaluation/inference only**; saved `y_pred` in diagnostics and in `predict_lambda()` are capped. Training artifacts (e.g. internal trees) are **not** capped.

---

## D) Features (and leakage / instability)

**Exact feature columns (X_cols):**  
- Not hard-coded; they are **all panel columns not in the exclude set**. For the **sampled** weekly panel produced by `build_weekly_sampled_future_panel`, the panel has (after merges and helpers) at least:  
  `segment_length`, `ROAD_CLASS`, `FROM_INTERSECTION_ID`, `TO_INTERSECTION_ID`, `segment_centroid_lat`, `segment_centroid_lon`, `hour_of_day`, `day_of_week`, `is_weekend`, `month`, `season`, `is_missing_weather`.  
- So the current training uses **12 features** (or 10 if `FROM_INTERSECTION_ID` / `TO_INTERSECTION_ID` are missing). Logs report "12 features".

**Lag/rolling crash-history features:**  
- **None** in the **sampled** training panel. The sampled panel is built from static + temporal + optional weather only; it does **not** add `past_crash_count_*` or `rolling_mean_7d` / `rolling_max_30d`.  
- The **full** panel from `build_panel_dataset()` does include lag/rolling (shift-then-roll, past-only), but that full panel is used for the backend snapshot and **not** for training the temporal count model. So **currently no lag/rolling in the training feature set**.

**Weather:**  
- **Not** included in practice. `build_weekly_sampled_future_panel(..., weather_data=None)` is called; `_attach_weather_features(panel, weather_data=None)` only adds `is_missing_weather=True`. So no real weather covariates; aggregation level N/A.

**Heavy-tailed / instability:**  
- **segment_length** can be long-tailed (e.g. long segments).  
- **No** clipping or winsorization of features in the current path; only `fillna(0)` and `astype(float)`. StandardScaler is applied to X before fit/predict, so features are scaled but not clipped.  
- **ROAD_CLASS** is categorical (e.g. "Ave", "Dr"); it is passed through and coerced to numeric (via `pd.to_numeric(..., errors="coerce").fillna(0)`), which can produce 0 or arbitrary codes. So there is some risk of unstable or non-portable encoding, but no explicit tail-weighting of features yet.

---

## E) What tail-weighting should accomplish (to be decided)

**Failure mode to fix (choose/prioritize):**  
- **Underpredicting large counts (2+ crashes/week)** — the “stripes” in the predicted-vs-actual plot (high actual, low predicted).  
- Optionally: **better top-K lift** or **better calibration in the tail**; these are not yet formalized as targets.

**Definition of “tail” for weighting:**  
- Not yet defined in code. Candidate thresholds: **y ≥ 2**, **y ≥ 3**, or **y ≥ 5** (counts per segment-week).  
- **Distribution counts:** From the last run, test set had mean 0.0876, 93.6% zeros, 6.4% positive; max 16. Exact counts at y≥2, y≥3, y≥5 are not currently logged; they can be added to `run_temporal_model_diagnostics.py` (e.g. print `(y_test >= 2).sum()`, etc.).

**Ranking vs exact counts:**  
- **Routing use case** favors **ranking** high-risk windows (so “safer route” avoids high-λ segments).  
- **Exact count matching** in the tail is secondary but may be desired for calibration or reporting; not yet specified.

---

## F) Evaluation and verification

**Diagnostics dataset:**  
- **Same as the model’s test set:** the **sampled** weekly panel, temporally split; test portion is the last 20% of unique `window_start` values. Diagnostics load `temporal_model_test_results.npz` (y_test, y_pred saved at training time). So diagnostics are on the **sampled** test set, **not** a full population-like test set.

**Weighted metrics:**  
- **No.** MAE, RMSE, Poisson deviance, correlation, binned calibration, and top-K lift are all computed **without** sample weights. So we do **not** currently compute weighted MAE or weighted lift.

**Lift/calibration with sampling weights:**  
- **No.** Lift is “mean(actual) in top 1%/5%/10% by predicted” on the raw test set; calibration is binned mean(actual) vs mean(predicted) per bin, unweighted.

**Acceptance criteria after tail-weighting:**  
- **Not yet defined.** Example to adopt: “Improve recall for y≥2 within top 1% predicted (or top-K lift for y≥2), while keeping binned calibration (e.g. top bin) within a reasonable range and not degrading overall correlation.”

---

## G) Guardrails

**Regularization:**  
- **max_depth=6**, **learning_rate=0.1**, **max_iter=300**.  
- **Not** set: `min_samples_leaf`, `l2_regularization`, `early_stopping` for the temporal Poisson HGBR.

**lambda_cap:**  
- **50.0** (crashes per segment-week). Set in `train_temporal_model.py` as `lambda_cap=50.0` and stored in the saved model; `predict_lambda()` and evaluation both clip to `[0, lambda_cap]`.  
- **How often hit:** From diagnostics, p99.9 of y_pred = 50 (many at cap); “Counts: >10: 345” and no >100, so about 345 rows at the cap (top 0.1% or so of test set).

**Feature clipping / standardization:**  
- **No** percentile clipping or winsorization of features.  
- **StandardScaler** is applied to X (fit on train, transform on train/val/test and at inference). No per-feature cap.

---

## H) Practical constraints

**Model:**  
- Tail-weighting is expected to work with the **current Poisson HistGradientBoostingRegressor** for now. A switch to NB (or hurdle) is not implemented yet.

**Scope of tail-weighting:**  
- **To be decided.** Options:  
  - **Training only:** Up-weight or reweight samples with y ≥ 2 (or y ≥ 3, etc.) in the training objective via `sample_weight` (e.g. increase weight for tail rows).  
  - **Evaluation only:** Report weighted metrics or tail-specific lift/recall in addition to current metrics.  
  - **Routing inference:** No change to how λ is used in routing; λ remains the capped model output (per segment-week), possibly improved by tail-weighting at training time.

---

## Summary table

| Item | Current state |
|------|----------------|
| Panel | Case-control: all positives + 10× sampled negatives (uniform over segment×week) |
| sample_weight | Yes: w0 = approx_zero_pairs / n_zero_sampled for zeros; 1 for positives |
| Target | future_crash_count (int, next week); window/horizon 168h / 168h |
| Model output | μ per segment-week (capped at 50 at eval/inference only) |
| Model | HistGradientBoostingRegressor(loss="poisson", max_depth=6, lr=0.1, max_iter=300) |
| sample_weight in fit | Yes, aligned to train rows |
| Features | 12: static (length, class, intersections, centroids) + temporal (hour, dow, weekend, month, season) + is_missing_weather; no lags, no weather data |
| Lag/rolling | None in current training panel |
| Diagnostics | Unweighted; on same sampled test set |
| lambda_cap | 50; hit by ~0.1% of test rows |
| Tail-weighting | Not implemented; goal and “tail” definition (y≥2?) to be decided |
