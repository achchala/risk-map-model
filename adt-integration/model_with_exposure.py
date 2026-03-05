"""
model_with_exposure.py
----------------------
Refits Poisson, Negative Binomial, and XGBoost on the merged dataset
using absolute exposure (volume × length) as the model offset.

Pipeline
--------
1.  Load model_dataset.csv
2.  Compute exposure = avg_daily_vol * segment_length
3.  Drop / report any zero-exposure rows  (log(0) = -inf)
4.  Compute log_exposure = log(exposure)
5.  Impute missing speed values with the column median
6.  Train / test split (stratified on zero vs non-zero crash)
7.  Fit Poisson GLM       — offset=log_exposure
8.  Fit NegBin GLM (NB2)  — offset=log_exposure
9.  Fit XGBoost           — base_margin=log_exposure
10. Cameron-Trivedi overdispersion stress test
11. Poisson zero-mass stress test
12. Out-of-sample metrics + verdict

Run from this directory:
    python model_with_exposure.py
"""

import warnings
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xgboost as xgb
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Poisson, NegativeBinomial
from statsmodels.tools import add_constant
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pathlib import Path

warnings.filterwarnings("ignore")

DATA_PATH  = Path(__file__).parent / "model_dataset.csv"
OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

DIV = "=" * 65
SEC = "-" * 65

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Load
# ─────────────────────────────────────────────────────────────────────────────
print(DIV)
print("STEP 1 — Load model_dataset.csv")
print(DIV)

df = pd.read_csv(DATA_PATH)
print(f"  Loaded {len(df):,} rows × {df.shape[1]} columns")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Compute absolute exposure
# ─────────────────────────────────────────────────────────────────────────────
print()
print(DIV)
print("STEP 2 — Compute exposure = avg_daily_vol × segment_length")
print(DIV)

df["exposure"] = df["avg_daily_vol"] * df["segment_length"]
print(f"  exposure range : [{df['exposure'].min():,.1f}, {df['exposure'].max():,.1f}]")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Handle zero exposure  (cannot take log of 0)
# ─────────────────────────────────────────────────────────────────────────────
print()
print(DIV)
print("STEP 3 — Handle zero/negative exposure")
print(DIV)

zero_exp = (df["exposure"] <= 0).sum()
null_exp  = df["exposure"].isna().sum()

if zero_exp == 0 and null_exp == 0:
    print(f"  [PASS] No zero or null exposure values — log() is safe.")
else:
    print(f"  [WARN] {zero_exp} zero/negative  +  {null_exp} null exposure rows — dropping.")
    df = df[(df["exposure"] > 0) & df["exposure"].notna()].copy()
    print(f"  Remaining: {len(df):,} rows")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Log-exposure  (the offset term)
# ─────────────────────────────────────────────────────────────────────────────
print()
print(DIV)
print("STEP 4 — Compute log_exposure  (GLM offset / XGBoost base_margin)")
print(DIV)

df["log_exposure"] = np.log(df["exposure"])
print(f"  log_exposure range : [{df['log_exposure'].min():.3f}, {df['log_exposure'].max():.3f}]")
assert df["log_exposure"].isna().sum() == 0
assert np.isfinite(df["log_exposure"]).all()
print(f"  [PASS] No NaN or Inf in log_exposure.")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Build feature matrix  (impute missing speeds with column median)
# ─────────────────────────────────────────────────────────────────────────────
print()
print(DIV)
print("STEP 5 — Build feature matrix  (impute missing speed with median)")
print(DIV)

SPEED_COLS = ["avg_speed", "avg_85th_percentile_speed", "avg_95th_percentile_speed"]
FEATURE_COLS = [c for c in SPEED_COLS if c in df.columns]

for col in FEATURE_COLS:
    n_missing = df[col].isna().sum()
    median    = df[col].median()
    if n_missing:
        df[col] = df[col].fillna(median)
        print(f"  Imputed {n_missing:,} missing values in {col} → median={median:.2f}")
    else:
        print(f"  {col}: no missing values  (median={median:.2f})")

# Y, X, offset arrays
y            = df["crash_count"].values.astype(int)
X_raw        = df[FEATURE_COLS].values.astype(float)
log_exposure = df["log_exposure"].values.astype(float)

print(f"\n  Feature matrix shape : {X_raw.shape}")
print(f"  Predictor columns    : {FEATURE_COLS}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — Train / test split  (stratified on zero vs non-zero)
# ─────────────────────────────────────────────────────────────────────────────
print()
print(DIV)
print("STEP 6 — Train / test split  (80 / 20, stratified)")
print(DIV)

strat = (y > 0).astype(int)
idx   = np.arange(len(y))
idx_tr, idx_te = train_test_split(idx, test_size=0.2, random_state=42, stratify=strat)

X_tr, X_te = X_raw[idx_tr], X_raw[idx_te]
y_tr, y_te = y[idx_tr],     y[idx_te]
off_tr      = log_exposure[idx_tr]
off_te      = log_exposure[idx_te]

print(f"  Train : {len(y_tr):,}  (zero rate {(y_tr==0).mean():.1%})")
print(f"  Test  : {len(y_te):,}  (zero rate {(y_te==0).mean():.1%})")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — Poisson GLM  with offset
# ─────────────────────────────────────────────────────────────────────────────
print()
print(DIV)
print("STEP 7 — Poisson GLM  (offset = log_exposure)")
print(DIV)

X_tr_sm = add_constant(X_tr, has_constant="add")
X_te_sm = add_constant(X_te, has_constant="add")

pois_result = None
try:
    pois_model = Poisson(y_tr, X_tr_sm, offset=off_tr).fit(disp=False, maxiter=300)
    mu_tr_pois = pois_model.predict(X_tr_sm, offset=off_tr)
    mu_te_pois = pois_model.predict(X_te_sm, offset=off_te)
    pois_result = {"pred_test": mu_te_pois, "aic": pois_model.aic,
                   "bic": pois_model.bic, "llf": pois_model.llf,
                   "ll_obs": pois_model.model.loglikeobs(np.asarray(pois_model.params))}
    print(f"  AIC={pois_model.aic:.1f}   BIC={pois_model.bic:.1f}   LogLik={pois_model.llf:.1f}")
    print(f"  Coefficients:")
    col_names = ["const"] + FEATURE_COLS
    for name, coef, pval in zip(col_names,
                                 pois_model.params,
                                 pois_model.pvalues):
        sig = "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else ""))
        print(f"    {name:<32s}  β={coef:+.4f}  p={pval:.4f} {sig}")
except Exception as e:
    print(f"  Poisson FAILED: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 8 — Negative Binomial GLM  with offset
# ─────────────────────────────────────────────────────────────────────────────
print()
print(DIV)
print("STEP 8 — Negative Binomial NB2  (offset = log_exposure)")
print(DIV)

nb_result = None
try:
    nb_model = NegativeBinomial(
        y_tr, X_tr_sm, loglike_method="nb2", offset=off_tr
    ).fit(disp=False, maxiter=400, method="bfgs")

    mu_te_nb  = nb_model.predict(X_te_sm, offset=off_te)
    params_nb = np.asarray(nb_model.params)

    # alpha (overdispersion) is the last parameter
    alpha_nb = float(params_nb[-1])

    nb_result = {"pred_test": mu_te_nb, "aic": nb_model.aic,
                 "bic": nb_model.bic, "llf": nb_model.llf,
                 "alpha": alpha_nb,
                 "ll_obs": nb_model.model.loglikeobs(params_nb)}

    print(f"  AIC={nb_model.aic:.1f}   BIC={nb_model.bic:.1f}   LogLik={nb_model.llf:.1f}")
    print(f"  Overdispersion  α = {alpha_nb:.4f}  "
          f"({'overdispersed — NB justified' if alpha_nb > 0.05 else 'near-equidispersed'})")
    print(f"  Coefficients:")
    col_names = ["const"] + FEATURE_COLS + ["alpha"]
    for name, coef, pval in zip(col_names,
                                 nb_model.params,
                                 nb_model.pvalues):
        sig = "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else ""))
        print(f"    {name:<32s}  β={coef:+.4f}  p={pval:.4f} {sig}")
except Exception as e:
    print(f"  NegBin FAILED: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 9 — XGBoost  with base_margin = log_exposure
# ─────────────────────────────────────────────────────────────────────────────
print()
print(DIV)
print("STEP 9 — XGBoost (count:poisson, base_margin = log_exposure)")
print(DIV)

xgb_result = None
try:
    dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=FEATURE_COLS)
    dtrain.set_base_margin(off_tr)          # <-- log_exposure as offset

    dtest  = xgb.DMatrix(X_te, label=y_te, feature_names=FEATURE_COLS)
    dtest.set_base_margin(off_te)

    params = {
        "objective":        "count:poisson",
        "max_depth":        5,
        "eta":              0.05,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "seed":             42,
        "verbosity":        0,
    }
    xgb_model = xgb.train(
        params, dtrain, num_boost_round=500,
        evals=[(dtest, "test")],
        early_stopping_rounds=30,
        verbose_eval=False,
    )
    mu_te_xgb = xgb_model.predict(dtest)
    xgb_result = {"pred_test": mu_te_xgb}

    importance = xgb_model.get_score(importance_type="gain")
    print(f"  Best round : {xgb_model.best_iteration}")
    print(f"  Feature importance (gain): {importance}")
except Exception as e:
    print(f"  XGBoost FAILED: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 10 — STRESS TEST A: Cameron-Trivedi overdispersion test
# ─────────────────────────────────────────────────────────────────────────────
print()
print(DIV)
print("STRESS TEST A — Cameron-Trivedi overdispersion  (on Poisson residuals)")
print(DIV)
print("  H0: equidispersion  (Poisson OK, α = 0)")
print("  H1: overdispersion  (NB needed,  α > 0)")

if pois_result is not None:
    mu_ct = pois_result["pred_test"]
    y_ct  = y_te

    lhs = (y_ct - mu_ct) ** 2 - y_ct
    rhs = mu_ct ** 2
    alpha_hat = np.dot(rhs, lhs) / np.dot(rhs, rhs)
    resid     = lhs - alpha_hat * rhs
    se_alpha  = np.sqrt(np.sum(resid**2) / (len(lhs) - 1)) / np.sqrt(np.dot(rhs, rhs))
    t_stat    = alpha_hat / se_alpha if se_alpha > 1e-15 else 0.0
    p_val     = 1 - stats.t.cdf(t_stat, df=len(lhs) - 1)

    print(f"\n  α̂  = {alpha_hat:.4f}")
    print(f"  t  = {t_stat:.3f}")
    print(f"  p  = {p_val:.6f}  (one-sided)")
    if p_val < 0.001:
        print("  [REJECT H0] *** Severe overdispersion — Poisson is WRONG. Use NB.")
    elif p_val < 0.05:
        print("  [REJECT H0] *   Overdispersion detected — NB preferred over Poisson.")
    else:
        print("  [FAIL TO REJECT H0] No significant overdispersion detected.")
else:
    print("  (Poisson did not fit — skipping)")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 11 — STRESS TEST B: Poisson zero-mass check
# ─────────────────────────────────────────────────────────────────────────────
print()
print(DIV)
print("STRESS TEST B — Poisson zero-mass check")
print(DIV)
print("  If observed zeros >> Poisson-expected zeros → zero inflation confirmed")

n_te           = len(y_te)
actual_zeros   = int((y_te == 0).sum())
mean_y_te      = y_te.mean()
poisson_exp_z  = int(np.round(n_te * np.exp(-mean_y_te)))
excess         = actual_zeros - poisson_exp_z

print(f"\n  Test-set size          : {n_te:,}")
print(f"  Mean crash count       : {mean_y_te:.4f}")
print(f"  Observed zeros         : {actual_zeros:,}  ({actual_zeros/n_te:.1%})")
print(f"  Poisson-expected zeros : {poisson_exp_z:,}  ({poisson_exp_z/n_te:.1%})")
print(f"  Excess zeros           : {excess:,}  ({excess/n_te:.1%})")

if excess > 0:
    print(f"  [CONFIRMED] Zero inflation — observed zeros exceed Poisson prediction.")
    if actual_zeros / n_te > 0.80:
        print(f"  Zero rate = {actual_zeros/n_te:.1%} → ZINB or Hurdle NB warranted.")
else:
    print(f"  [PASS] No excess zeros — standard Poisson/NB adequate for zero mass.")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 12 — Out-of-sample metrics
# ─────────────────────────────────────────────────────────────────────────────
print()
print(DIV)
print("STEP 12 — Out-of-sample metrics  (test set)")
print(DIV)

def mpd(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-9, None)
    term = np.where(y_true > 0,
                    y_true * np.log(y_true / y_pred) - (y_true - y_pred),
                    -(y_true - y_pred))
    return 2.0 * term.mean()

model_preds = {}
if pois_result: model_preds["Poisson"]  = pois_result["pred_test"]
if nb_result:   model_preds["NegBin"]   = nb_result["pred_test"]
if xgb_result:  model_preds["XGBoost"]  = xgb_result["pred_test"]

rows = []
for name, y_pred in model_preds.items():
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0, None)
    rows.append({
        "Model": name,
        "MAE":          round(mean_absolute_error(y_te, y_pred), 4),
        "RMSE":         round(np.sqrt(mean_squared_error(y_te, y_pred)), 4),
        "MeanPoisDev":  round(mpd(y_te, y_pred), 4),
        "Pred_zeros":   int((y_pred < 0.5).sum()),
        "Actual_zeros": int((y_te == 0).sum()),
    })

results_df = pd.DataFrame(rows).set_index("Model")
print(f"\n{results_df.to_string()}")

# AIC / BIC comparison (parametric models only)
print(f"\n  AIC / BIC:")
for name, r in [("Poisson", pois_result), ("NegBin", nb_result)]:
    if r:
        print(f"    {name:<10}  AIC={r['aic']:.1f}   BIC={r['bic']:.1f}")

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
print()
print(DIV)
print("SUMMARY")
print(DIV)

if nb_result and pois_result:
    aic_drop = pois_result["aic"] - nb_result["aic"]
    print(f"  AIC(Poisson) - AIC(NB) = {aic_drop:.1f}")
    if aic_drop > 10:
        print(f"  NB is decisively better (ΔAIC > 10). Exposure offset + NB is the right model.")
    elif aic_drop > 2:
        print(f"  NB is moderately better (ΔAIC={aic_drop:.1f}). Prefer NB.")
    else:
        print(f"  Models are comparable (ΔAIC={aic_drop:.1f}).")

print(f"\n  Datasets used  : {DATA_PATH.name}")
print(f"  Exposure term  : avg_daily_vol × segment_length  (passed as log-offset)")
print(f"  Key insight    : crash rate is now normalised by traffic × distance,")
print(f"                   not confounded by segment length or volume alone.")

# Save results
out_csv = OUTPUT_DIR / "model_results_with_exposure.csv"
results_df.to_csv(out_csv)
print(f"\n  Results saved → {out_csv}")
