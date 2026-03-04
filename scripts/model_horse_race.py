"""
Model Horse Race: Validating the Professor's Claim
===================================================
Tests whether Poisson is defensible against:
  1. Negative Binomial      — handles overdispersion
  2. Zero-Inflated NB       — handles structural zeros
  3. XGBoost                — captures non-linear interactions

Key diagnostics run BEFORE fitting any model:
  - Zero rate (% of Y == 0)
  - Dispersion ratio: Var(Y) / E(Y)  (Poisson assumes this == 1)
  - Cameron & Trivedi overdispersion t-test
  - Observed vs Poisson-expected zero count

Then compares all four models on:
  - AIC / BIC (parametric models only)
  - Vuong test: Poisson vs NB, NB vs ZINB
  - Out-of-sample: MAE, RMSE, Mean Poisson Deviance
  - Zero-prediction accuracy: does the model get zeros right?
  - Rootogram (visual fit of predicted count distribution)

Usage:
    python scripts/model_horse_race.py
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import (
    Poisson,
    NegativeBinomial,
)
from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP
from statsmodels.tools import add_constant

warnings.filterwarnings("ignore")

# ── project paths ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_processing.data_loader import load_and_clean_data
from src.data_processing.spatial_join_fast import perform_spatial_join_fast
from src.feature_engineering.feature_creator import create_segment_features

# ── helpers ───────────────────────────────────────────────────────────────────

DIVIDER = "=" * 70
SECTION  = "-" * 70

def _mean_poisson_deviance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Poisson deviance: 2 * mean[ y*log(y/ŷ) - (y - ŷ) ]
    Lower is better; Poisson regression minimises this exactly.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 1e-9, None)
    term = np.where(
        y_true > 0,
        y_true * np.log(y_true / y_pred) - (y_true - y_pred),
        -(y_true - y_pred),
    )
    return 2.0 * term.mean()


def _zero_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    How well does the model predict zeros vs non-zeros?
    Threshold: predict zero if ŷ < 0.5, else predict non-zero.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    predicted_zero = y_pred < 0.5
    actual_zero    = y_true == 0
    tp = np.sum(predicted_zero & actual_zero)      # correctly called zero
    fp = np.sum(predicted_zero & ~actual_zero)     # wrongly called zero
    fn = np.sum(~predicted_zero & actual_zero)     # missed a zero
    tn = np.sum(~predicted_zero & ~actual_zero)    # correctly called non-zero
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    return {
        "zero_precision": precision,
        "zero_recall":    recall,
        "predicted_zeros":  int(predicted_zero.sum()),
        "actual_zeros":     int(actual_zero.sum()),
    }


def _vuong_test(ll1: np.ndarray, ll2: np.ndarray, n: int) -> tuple:
    """
    Vuong (1989) non-nested test of H0: models are equally close to DGP.
    Returns (z_stat, p_value).
    Positive z favours model 1; negative z favours model 2.
    """
    m = ll1 - ll2
    mean_m = m.mean()
    std_m  = m.std(ddof=1)
    if std_m < 1e-15:
        return 0.0, 1.0
    z = (np.sqrt(n) * mean_m) / std_m
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return float(z), float(p)


def _cameron_trivedi_test(y: np.ndarray, mu: np.ndarray) -> tuple:
    """
    Cameron & Trivedi (1990) auxiliary OLS overdispersion test.
    Regress (y - ŷ)² - y  on  ŷ²  (NB2 alternative) using OLS.
    H0: coefficient α = 0  (equidispersion, Poisson OK)
    H1: α > 0  (overdispersion, NB better)
    Returns (alpha_hat, t_stat, p_value).
    """
    lhs = (y - mu) ** 2 - y
    rhs = mu ** 2
    # OLS without intercept
    cov_xy = np.dot(rhs, lhs)
    var_x  = np.dot(rhs, rhs)
    if var_x < 1e-15:
        return 0.0, 0.0, 1.0
    alpha_hat = cov_xy / var_x
    resid     = lhs - alpha_hat * rhs
    se_alpha  = np.sqrt(np.sum(resid ** 2) / (len(lhs) - 1)) / np.sqrt(var_x)
    t_stat    = alpha_hat / se_alpha if se_alpha > 1e-15 else 0.0
    p_value   = 1 - stats.t.cdf(t_stat, df=len(lhs) - 1)   # one-sided, H1: α > 0
    return float(alpha_hat), float(t_stat), float(p_value)


def _plot_rootogram(y_true, predictions: dict, out_path: Path):
    """
    Hanging rootogram: sqrt(observed) - sqrt(expected) per count value k.
    A good fit hovers near zero.  A systematic gap at k=0 = zero-inflation.
    """
    counts = np.arange(0, min(int(y_true.max()) + 1, 40))
    fig, axes = plt.subplots(1, len(predictions), figsize=(6 * len(predictions), 5))
    if len(predictions) == 1:
        axes = [axes]

    observed_freq = np.array([(y_true == k).sum() for k in counts])

    for ax, (name, y_pred) in zip(axes, predictions.items()):
        expected_freq = np.array([y_pred[y_pred.round(0) == k].shape[0] for k in counts])
        # Simpler: bin the continuous predictions
        expected_freq = np.histogram(y_pred, bins=np.arange(-0.5, counts[-1] + 1.5))[0][:len(counts)]
        # hanging bars
        gap = np.sqrt(observed_freq) - np.sqrt(expected_freq)
        colours = ["#d73027" if g < 0 else "#4575b4" for g in gap]
        ax.bar(counts, gap, color=colours, edgecolor="black", linewidth=0.5)
        ax.axhline(0, color="black", linewidth=1)
        ax.set_title(f"{name}\n√(obs) − √(exp)")
        ax.set_xlabel("Crash count k")
        ax.set_ylabel("Hanging difference")

    plt.suptitle("Rootogram: Model Fit by Count Value\n"
                 "Blue bars = over-predicted, Red bars = under-predicted",
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Rootogram saved → {out_path}")


# ── main pipeline ─────────────────────────────────────────────────────────────

def build_dataset():
    """Run the existing pipeline and return a clean feature matrix."""
    print(f"\n{DIVIDER}")
    print("STEP 1 — Loading data via existing pipeline")
    print(DIVIDER)
    data_dir = PROJECT_ROOT / "data"
    collision_data, ksi_data, road_network = load_and_clean_data(data_dir)
    segment_crashes = perform_spatial_join_fast(collision_data, ksi_data, road_network)
    features = create_segment_features(segment_crashes, road_network)
    print(f"  Total road segments : {len(features):,}")
    return features


def select_predictors(features: pd.DataFrame) -> tuple:
    """
    Returns (X, y) using structural road features as predictors.

    Using crash counts as predictors in count regression is circular.
    We use:
        - segment_length
        - road_class dummies from the actual ROAD_CLASS / LINEAR_NAME_TYPE
          field (e.g. Ave, St, Blvd, Rd, Trl ...) since the pre-built
          arterial/collector dummies are all-zero in this dataset
          (LINEAR_NAME_TYPE stores street suffixes, not functional class).

    Any column with zero variance is dropped before returning.
    """
    X_df = features[["segment_length"]].copy().fillna(0)

    # Encode actual road suffix type (LINEAR_NAME_TYPE / ROAD_CLASS)
    road_col = None
    for candidate in ["ROAD_CLASS", "LINEAR_NAME_TYPE"]:
        if candidate in features.columns:
            road_col = candidate
            break

    if road_col is not None:
        # Keep the top-N most common categories; lump rest as "Other"
        top_n = 8
        top_cats = features[road_col].value_counts().nlargest(top_n).index.tolist()
        rc = features[road_col].where(features[road_col].isin(top_cats), other="Other")
        dummies = pd.get_dummies(rc, prefix="rc", drop_first=True)
        X_df = pd.concat([X_df, dummies], axis=1)

    # Drop any column with zero variance (would cause singular matrix)
    var_mask = X_df.var(ddof=0) > 0
    dropped = list(X_df.columns[~var_mask])
    if dropped:
        print(f"  [select_predictors] Dropping zero-variance columns: {dropped}")
    X_df = X_df.loc[:, var_mask]

    predictor_cols = list(X_df.columns)
    y = features["num_total_crashes"].fillna(0).astype(int).values
    X = X_df.values.astype(float)

    print(f"  [select_predictors] Using {len(predictor_cols)} predictors: {predictor_cols}")
    return X, y, predictor_cols


def run_diagnostics(y: np.ndarray):
    """Section 2 — data-level diagnostics BEFORE any model is fit."""
    print(f"\n{DIVIDER}")
    print("STEP 2 — Target variable diagnostics (Y = crash count per segment)")
    print(DIVIDER)

    n          = len(y)
    zero_rate  = (y == 0).mean()
    mean_y     = y.mean()
    var_y      = y.var(ddof=1)
    disp_ratio = var_y / mean_y if mean_y > 0 else float("inf")

    print(f"  N segments            : {n:,}")
    print(f"  Zero-crash segments   : {(y==0).sum():,}  ({zero_rate:.1%})")
    print(f"  Non-zero segments     : {(y>0).sum():,}")
    print(f"  Mean  E[Y]            : {mean_y:.4f}")
    print(f"  Variance  Var[Y]      : {var_y:.4f}")
    print(f"  Dispersion  Var/Mean  : {disp_ratio:.2f}x")

    # Poisson-expected zeros: sum_i exp(-mu_hat) where mu_hat = mean_y (intercept-only)
    poisson_expected_zeros = int(np.round(n * np.exp(-mean_y)))
    actual_zeros           = int((y == 0).sum())
    print(f"\n  Under a Poisson(mean={mean_y:.4f}) with N={n:,} observations:")
    print(f"    Expected zeros     : {poisson_expected_zeros:,}")
    print(f"    Observed zeros     : {actual_zeros:,}")
    excess = actual_zeros - poisson_expected_zeros
    print(f"    Excess zeros       : {excess:,}  "
          f"({'ZERO INFLATION CONFIRMED' if excess > 0 else 'no excess'})")

    print(f"\n  VERDICT:")
    if disp_ratio > 2:
        print(f"    Var/Mean = {disp_ratio:.1f}x >> 1  →  SEVERE overdispersion.  Poisson is wrong.")
    elif disp_ratio > 1.2:
        print(f"    Var/Mean = {disp_ratio:.1f}x > 1  →  Moderate overdispersion.  NB likely needed.")
    else:
        print(f"    Var/Mean = {disp_ratio:.2f}  ≈ 1  →  Equidispersion plausible.  Poisson may be OK.")

    if zero_rate > 0.9:
        print(f"    Zero rate = {zero_rate:.1%}  →  EXTREME zero inflation.  ZINB / Hurdle warranted.")
    elif zero_rate > 0.7:
        print(f"    Zero rate = {zero_rate:.1%}  →  HIGH zero inflation.  Consider ZINB.")

    return {
        "n": n, "zero_rate": zero_rate, "mean_y": mean_y,
        "var_y": var_y, "disp_ratio": disp_ratio,
        "actual_zeros": actual_zeros,
        "poisson_expected_zeros": poisson_expected_zeros,
    }


def fit_models(X_train, y_train, X_test, y_test, predictor_cols):
    """Section 3 — fit all four models."""
    print(f"\n{DIVIDER}")
    print("STEP 3 — Fitting models")
    print(DIVIDER)

    # statsmodels needs a constant term for the intercept
    X_train_sm = add_constant(X_train, has_constant="add")
    X_test_sm  = add_constant(X_test,  has_constant="add")

    results = {}

    # ── 3a. Poisson ──────────────────────────────────────────────────────────
    print("\n  [1/4] Poisson regression ...")
    try:
        pois_model = Poisson(y_train, X_train_sm).fit(disp=False, maxiter=200)
        mu_train_pois = pois_model.predict(X_train_sm)
        mu_test_pois  = pois_model.predict(X_test_sm)

        # Cameron-Trivedi test on training residuals
        ct_alpha, ct_t, ct_p = _cameron_trivedi_test(y_train, mu_train_pois)

        ll_obs_pois = pois_model.model.loglikeobs(np.asarray(pois_model.params))
        results["Poisson"] = {
            "model":        pois_model,
            "pred_test":    mu_test_pois,
            "aic":          pois_model.aic,
            "bic":          pois_model.bic,
            "llf":          pois_model.llf,
            "ll_obs_train": ll_obs_pois,
            "ct_alpha":     ct_alpha,
            "ct_t":         ct_t,
            "ct_p":         ct_p,
        }
        print(f"    AIC={pois_model.aic:.1f}  BIC={pois_model.bic:.1f}  "
              f"LogLik={pois_model.llf:.1f}")
        print(f"    Cameron-Trivedi overdispersion: α={ct_alpha:.4f}  "
              f"t={ct_t:.2f}  p={ct_p:.4f}"
              f"  {'→ REJECT Poisson (overdispersed)' if ct_p < 0.05 else '→ fail to reject Poisson'}")
    except Exception as e:
        print(f"    Poisson failed: {e}")
        results["Poisson"] = None

    # ── 3b. Negative Binomial (NB2) ──────────────────────────────────────────
    print("\n  [2/4] Negative Binomial (NB2) ...")
    try:
        nb_model = NegativeBinomial(y_train, X_train_sm, loglike_method="nb2").fit(
            disp=False, maxiter=300, method="bfgs"
        )
        mu_test_nb = nb_model.predict(X_test_sm)
        # params may be a numpy array or pandas Series depending on statsmodels version
        params_nb = nb_model.params
        if hasattr(params_nb, "loc"):          # pandas Series
            alpha_nb = float(params_nb["alpha"])
        else:                                   # numpy array — alpha is last element
            alpha_nb = float(params_nb[-1])

        ll_obs_nb = nb_model.model.loglikeobs(np.asarray(params_nb))

        results["NegBin"] = {
            "model":        nb_model,
            "pred_test":    mu_test_nb,
            "aic":          nb_model.aic,
            "bic":          nb_model.bic,
            "llf":          nb_model.llf,
            "ll_obs_train": ll_obs_nb,
            "alpha":        alpha_nb,
        }
        print(f"    AIC={nb_model.aic:.1f}  BIC={nb_model.bic:.1f}  "
              f"LogLik={nb_model.llf:.1f}  α(dispersion)={alpha_nb:.4f}")
        if alpha_nb > 0.1:
            print(f"    α={alpha_nb:.4f} > 0  → NB fits overdispersion that Poisson ignores")
    except Exception as e:
        print(f"    NB failed: {e}")
        results["NegBin"] = None

    # ── 3c. Zero-Inflated Negative Binomial (ZINB) ───────────────────────────
    print("\n  [3/4] Zero-Inflated Negative Binomial (ZINB) ...")
    try:
        inflate_exog_train = np.ones((len(y_train), 1))
        inflate_exog_test  = np.ones((len(y_test),  1))

        # Build warm-start params from Poisson (count part) + inflation intercept
        # logit(zero_rate) gives a good inflation starting point
        zero_rate_train = (y_train == 0).mean()
        infl_intercept  = np.log(zero_rate_train / (1 - zero_rate_train + 1e-9))

        if results.get("Poisson") is not None:
            pois_params = np.asarray(results["Poisson"]["model"].params)
        else:
            pois_params = np.zeros(X_train_sm.shape[1])

        # ZINB params: [count_params..., log_alpha, infl_params...]
        start_params = np.concatenate([pois_params, [0.5], [infl_intercept]])

        zinb_model = ZeroInflatedNegativeBinomialP(
            y_train, X_train_sm, exog_infl=inflate_exog_train, p=2
        ).fit(
            start_params=start_params,
            disp=False, maxiter=500,
            method="bfgs",
            gtol=1e-4,
        )

        mu_test_zinb_raw = zinb_model.predict(X_test_sm, exog_infl=inflate_exog_test)
        # Clip to prevent infinities from exploding exp() during non-convergence
        mu_test_zinb = np.clip(mu_test_zinb_raw, 0, 1e6)

        converged = zinb_model.mle_retvals.get("converged", True)
        llf = zinb_model.llf
        ll_obs_zinb = zinb_model.model.loglikeobs(np.asarray(zinb_model.params))

        if np.isnan(llf):
            print(f"    ZINB converged={converged} but LogLik=NaN → non-convergence, skipping")
            results["ZINB"] = None
        else:
            results["ZINB"] = {
                "model":        zinb_model,
                "pred_test":    mu_test_zinb,
                "aic":          zinb_model.aic,
                "bic":          zinb_model.bic,
                "llf":          llf,
                "ll_obs_train": ll_obs_zinb,
            }
            print(f"    AIC={zinb_model.aic:.1f}  BIC={zinb_model.bic:.1f}  "
                  f"LogLik={llf:.1f}  converged={converged}")
    except Exception as e:
        print(f"    ZINB failed: {e}")
        results["ZINB"] = None

    # ── 3d. XGBoost ──────────────────────────────────────────────────────────
    print("\n  [4/4] XGBoost (count:poisson objective) ...")
    try:
        # Scale for XGBoost
        scaler = StandardScaler()
        X_train_xgb = scaler.fit_transform(X_train)
        X_test_xgb  = scaler.transform(X_test)

        dtrain = xgb.DMatrix(X_train_xgb, label=y_train, feature_names=predictor_cols)
        dtest  = xgb.DMatrix(X_test_xgb,  label=y_test,  feature_names=predictor_cols)

        params = {
            "objective":        "count:poisson",
            "max_depth":        6,
            "eta":              0.05,
            "subsample":        0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 5,
            "seed":             42,
            "verbosity":        0,
        }
        xgb_model = xgb.train(
            params, dtrain, num_boost_round=400,
            evals=[(dtrain, "train"), (dtest, "test")],
            early_stopping_rounds=30,
            verbose_eval=False,
        )
        mu_test_xgb = xgb_model.predict(dtest)

        # Feature importance
        importance = xgb_model.get_fscore()

        results["XGBoost"] = {
            "model":       xgb_model,
            "pred_test":   mu_test_xgb,
            "aic":         None,  # not defined for tree models
            "bic":         None,
            "llf":         None,
            "importance":  importance,
        }
        print(f"    Best round: {xgb_model.best_iteration}  "
              f"[no AIC/BIC — tree model]")
        print(f"    Feature importance: {importance}")
    except Exception as e:
        print(f"    XGBoost failed: {e}")
        results["XGBoost"] = None

    return results


def evaluate_models(results: dict, y_test: np.ndarray, y_train: np.ndarray):
    """Section 4 — out-of-sample metrics + Vuong tests."""
    print(f"\n{DIVIDER}")
    print("STEP 4 — Out-of-sample evaluation on hold-out test set")
    print(DIVIDER)

    rows = []
    preds = {}
    for name, r in results.items():
        if r is None:
            continue
        y_pred = np.asarray(r["pred_test"], dtype=float)
        # Skip models whose predictions contain NaN or Inf (non-convergence)
        if not np.isfinite(y_pred).all():
            bad = (~np.isfinite(y_pred)).sum()
            print(f"  Skipping {name}: {bad} non-finite predictions (model did not converge)")
            continue
        preds[name] = y_pred

        mae   = mean_absolute_error(y_test, y_pred)
        rmse  = np.sqrt(mean_squared_error(y_test, y_pred))
        mpd   = _mean_poisson_deviance(y_test, y_pred)
        za    = _zero_accuracy(y_test, y_pred)

        rows.append({
            "Model":         name,
            "AIC":           f"{r['aic']:.1f}" if r["aic"] else "—",
            "BIC":           f"{r['bic']:.1f}" if r["bic"] else "—",
            "MAE":           round(mae,  4),
            "RMSE":          round(rmse, 4),
            "MeanPoisDev":   round(mpd,  4),
            "Zero_Recall":   f"{za['zero_recall']:.1%}",
            "Zero_Precision":f"{za['zero_precision']:.1%}",
            "Pred_Zeros":    za["predicted_zeros"],
            "Actual_Zeros":  za["actual_zeros"],
        })

    df = pd.DataFrame(rows).set_index("Model")
    print(f"\n  {df.to_string()}")

    # ── Vuong tests ──────────────────────────────────────────────────────────
    print(f"\n{SECTION}")
    print("  Vuong Non-Nested Model Tests  (positive z favours model A)")
    print(SECTION)

    def _get_ll_obs(name):
        """Point-wise log-likelihoods on training data."""
        r = results.get(name)
        if r and "ll_obs_train" in r:
            return r["ll_obs_train"]
        return None

    pairs = [
        ("Poisson", "NegBin",  "Poisson vs NB:    +z → Poisson better"),
        ("NegBin",  "ZINB",    "NB     vs ZINB:   +z → NB better"),
        ("Poisson", "ZINB",    "Poisson vs ZINB:  +z → Poisson better"),
    ]
    for m1, m2, label in pairs:
        ll1 = _get_ll_obs(m1)
        ll2 = _get_ll_obs(m2)
        if ll1 is None or ll2 is None:
            print(f"  {label}  →  (skipped — model not available)")
            continue
        # Ensure same length
        n = min(len(ll1), len(ll2))
        z, p = _vuong_test(ll1[:n], ll2[:n], n)
        winner = m1 if z > 0 else m2
        sig    = "p<0.05 SIGNIFICANT" if p < 0.05 else "not significant"
        print(f"  {label}  →  z={z:.3f}  p={p:.4f}  ({sig})"
              f"  Best: {winner}")

    return df, preds


def print_verdict(diag: dict, eval_df: pd.DataFrame):
    """Section 5 — plain-language verdict for the defence."""
    print(f"\n{DIVIDER}")
    print("STEP 5 — VERDICT: Is the Professor Right?")
    print(DIVIDER)

    dr = diag["disp_ratio"]
    zr = diag["zero_rate"]
    ez = diag["poisson_expected_zeros"]
    az = diag["actual_zeros"]

    print(f"""
  DATA FACTS:
    Dispersion ratio  Var(Y)/E(Y) = {dr:.2f}x   (Poisson requires = 1.00)
    Zero rate                     = {zr:.1%}
    Poisson-expected zeros        = {ez:,}
    Observed zeros                = {az:,}
    Excess zeros                  = {az - ez:,}
""")

    if dr > 2:
        print("  CLAIM 1 — Overdispersion: CONFIRMED.")
        print(f"    Var(Y) = {dr:.1f}x * E(Y).  Poisson is mis-specified.")
        print("    A Cameron-Trivedi t-test above should also show p << 0.05.\n")
    else:
        print(f"  CLAIM 1 — Overdispersion: Weak ({dr:.2f}x).  Poisson might survive.\n")

    if zr > 0.9:
        print("  CLAIM 2 — Zero inflation: CONFIRMED.")
        print(f"    {zr:.1%} zeros >> what any Poisson(mean={diag['mean_y']:.4f}) can produce.")
        print("    ZINB or Hurdle NB is structurally required.\n")
    elif zr > 0.7:
        print(f"  CLAIM 2 — Zero inflation: LIKELY ({zr:.1%} zeros).")
        print("    Strong evidence for ZINB over standard NB.\n")

    if "XGBoost" in eval_df.index and "Poisson" in eval_df.index:
        mae_xgb = float(eval_df.loc["XGBoost", "MAE"])
        mae_pois = float(eval_df.loc["Poisson",  "MAE"])
        if mae_xgb < mae_pois * 0.9:
            pct = (1 - mae_xgb / mae_pois) * 100
            print(f"  CLAIM 3 — Non-linear structure: CONFIRMED.")
            print(f"    XGBoost MAE = {mae_xgb:.4f}  vs  Poisson MAE = {mae_pois:.4f}")
            print(f"    XGBoost is {pct:.1f}% more accurate out-of-sample.")
            print("    The professor is right: linear mean structure is wrong.\n")
        elif mae_xgb < mae_pois:
            print(f"  CLAIM 3 — Non-linear structure: MARGINAL.")
            print(f"    XGBoost edges Poisson ({mae_xgb:.4f} vs {mae_pois:.4f}) but not decisively.\n")
        else:
            print(f"  CLAIM 3 — Non-linear structure: NOT CONFIRMED on these features.")
            print(f"    Poisson MAE ({mae_pois:.4f}) ≤ XGBoost MAE ({mae_xgb:.4f}).")
            print("    Note: with richer features (speed limit, AADT) the gap may widen.\n")

    print("  SUMMARY:")
    print("    The professor's core claim stands whenever Var(Y)/E(Y) >> 1 and")
    print("    zero rate >> what Poisson predicts.  You cannot defend Poisson")
    print("    without first running these diagnostics and showing they pass.")
    print("    If they fail (as expected here), you must justify NB or ZINB")
    print("    with formal tests, and benchmark against XGBoost.\n")


def main():
    print(f"\n{DIVIDER}")
    print("  MODEL HORSE RACE — Poisson vs NB vs ZINB vs XGBoost")
    print(f"{DIVIDER}")

    # 1. Data
    features = build_dataset()

    # 2. Features & target
    X, y, predictor_cols = select_predictors(features)
    print(f"\n  Predictors used: {predictor_cols}")

    # 3. Diagnostics
    diag = run_diagnostics(y)

    # 4. Train/test split (stratified on zero vs non-zero to preserve zero rate)
    strat = (y > 0).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=strat
    )
    print(f"\n  Train: {len(y_train):,}  Test: {len(y_test):,}"
          f"  (zero rate  train={( y_train==0).mean():.1%}  "
          f"test={(y_test==0).mean():.1%})")

    # 5. Fit
    model_results = fit_models(X_train, y_train, X_test, y_test, predictor_cols)

    # 6. Evaluate
    eval_df, preds = evaluate_models(model_results, y_test, y_train)

    # 7. Rootogram
    out_dir = PROJECT_ROOT / "outputs" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{SECTION}")
    print("  STEP 6 — Rootogram (predicted vs observed count distribution)")
    print(SECTION)
    _plot_rootogram(y_test, preds, out_dir / "rootogram.png")

    # 8. Verdict
    print_verdict(diag, eval_df)

    # 9. Save table
    eval_df.to_csv(out_dir / "model_horse_race_results.csv")
    print(f"  Results table saved → {out_dir / 'model_horse_race_results.csv'}")


if __name__ == "__main__":
    main()
