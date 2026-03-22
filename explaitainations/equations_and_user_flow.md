# Equations & User Flow

## Hurdle Model Equations

---

**Stage 1 — Logit** `log(P/(1-P)) = f₁(X)`

Gradient-boosted classifier fits a log-loss objective over all windows. Output is a raw score converted to `P(crash_occurs)`.

- `P` = crash probability
- `f₁` = learned tree ensemble
- `X` = feature matrix

---

**Isotonic Calibration** `P_cal = isotonic(P_raw)`

Corrects the classifier's output so it reads as a true probability. Fitted on the held-out validation set to fix tree-classifier overconfidence.

- `P_cal` = calibrated probability
- `P_raw` = raw classifier score

---

**Stage 2 — Poisson Count** `log(E[y|crash]) = f₂(X)`

Poisson-loss regressor trained **only on crash windows**. Output is expected crash count given a crash occurred.

- `y` = crash count
- `f₂` = learned tree ensemble
- `X` = feature matrix

---

**Hurdle Combination** `λ = P(crash) × E[count|crash]`

Multiplies both stages into one expected crash rate per window per segment. Core prediction of the model.

- `λ` = expected crashes/window
- `P(crash)` = Stage 1 output
- `E[count|crash]` = Stage 2 output

---

**Tail Weight** `w_tail = 1 + α × log(1 + y)`

Upweights rare high-count windows so the optimizer doesn't ignore 5-crash events in a sea of 1-crash windows. Applied when `y ≥ 2`.

- `w_tail` = tail emphasis weight
- `α` = emphasis strength (2.0)
- `y` = crash count

---

**Final Sample Weight** `w_final = w_samp × w_tail`

Merges the sampling correction weight with tail emphasis. Normalized to preserve gradient scale, then capped at 50.

- `w_final` = combined training weight
- `w_samp` = sampling correction weight
- `w_tail` = tail weight

---

**Lambda Cap** `λ = clip(λ, 0, 50)`

Stability cap — prevents extreme predictions on rare segments from dominating routing math.

- `λ` = predicted crash rate

---

**Poisson Window Probability** `P_raw = 1 − e^{−λ}`

Converts raw λ into a window-level crash probability (Poisson CDF). Used as the calibration target.

- `P_raw` = uncalibrated crash probability
- `λ` = expected crashes/window

---

**Poisson Deviance** `D = 2 × mean(ŷ − y + y × log(y/ŷ))`

Evaluation metric. Measures how well the predicted count distribution matches actual counts.

- `D` = deviance score
- `ŷ` = predicted count
- `y` = actual count

---

## Routing Engine Equations

---

**Travel Time** `t = len_m / (speed_kmh / 3.6) / 3600`

Converts a segment's physical length and assumed road-class speed into hours. Base edge weight for the fastest route.

- `t` = travel time (hours)
- `len_m` = segment length (meters)
- `speed_kmh` = road-class speed

---

**GPS Snap Distance** `dist ≈ Δdeg × 111,000`

Flat-earth approximation to snap a GPS coordinate to the nearest graph node. Rejects snaps beyond 300 m.

- `dist` = approx distance (meters)
- `Δdeg` = degree difference in lat or lon

---

**Combined Multiplier** `m = m_weather × m_time`

Multiplies weather and time-of-day factors into one scalar applied to all λ values before routing.

- `m` = combined multiplier
- `m_weather` = weather factor
- `m_time` = time-of-day factor

---

**Adjusted Lambda** `λ_adj = λ × m`

Scales each segment's predicted crash rate up for hazardous conditions (e.g., snow + rush hour = 1.875×).

- `λ_adj` = adjusted crash rate
- `λ` = model-predicted crash rate
- `m` = combined multiplier

---

**Expected Crashes** `E = λ_adj × t`

Rate × time gives a dimensionless expected crash count for one traversal of the segment.

- `E` = expected crashes
- `λ_adj` = adjusted crash rate
- `t` = travel time (hours)

---

**Risk Weight** `w_risk = t + β × E`

Adds a crash penalty to travel time, making riskier segments appear "slower" to Dijkstra.

- `w_risk` = risk-penalized edge weight (hours)
- `t` = travel time
- `β` = penalty scalar (0.1 hrs/crash)
- `E` = expected crashes

---

**Route Total Expected Crashes** `Λ = Σ (λᵢ × tᵢ)`

Sums expected crashes across every edge in a route to get total trip-level crash exposure.

- `Λ` = total expected crashes
- `λᵢ` = adjusted crash rate for edge i
- `tᵢ` = travel time for edge i

---

**Route Crash Probability** `P = 1 − e^{−Λ}`

Poisson CDF: probability of at least one crash on the full route. Shown as `routeProbability` in the API response.

- `P` = crash probability
- `Λ` = total expected crashes

---

**Risk Label Thresholds** `λ ≤ p70 → low` / `p70 < λ ≤ p90 → medium` / `λ > p90 → high`

Assigns color labels relative to the city-wide λ distribution, not arbitrary fixed values.

- `λ` = segment crash rate
- `p70` = 70th percentile of all segments
- `p90` = 90th percentile of all segments

---

## User Flow in Plain English

**Before you open the app**, the server pre-loads the Toronto road network into a graph, assigns travel times to every edge based on road class, and runs the hurdle model to get a λ (crash rate) for every segment — all cached in memory.

**When you search for a route**, your origin and destination GPS points are snapped to the nearest graph nodes. The server then adjusts every segment's λ upward based on current weather and time of day.

**Two routes are computed** with Dijkstra simultaneously — one minimizing raw travel time, one minimizing a risk-penalized travel time where dangerous segments look artificially slower.

**Each route is scored** by summing λ × time across all its edges to get total expected crashes (Λ), then converting that to a crash probability with 1 − e^{−Λ}.

**The map renders** both routes with per-segment color coding (green/yellow/red) based on how each segment's λ compares to all of Toronto — so a "high risk" label means that segment is in the worst 10% city-wide, not just high in absolute terms.
