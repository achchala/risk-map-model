"""
Two-stage hurdle model for temporal crash prediction.

Stage 1 (binary): HistGradientBoostingClassifier — predicts P(crash occurs)
Stage 2 (count):  HistGradientBoostingRegressor(Poisson) — predicts
                  E[crash_count | crash occurs], trained only on positive windows.

Combined inference:
    λ_overall = P(crash_occurs) × E[crash_count | crash_occurs]

Stage 1 probabilities are isotonic-calibrated on the validation set.
Mirrors TemporalCountModelTrainer's interface for drop-in replacement.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import pickle
import sys
from typing import Tuple, Dict, Any, Optional

sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from config import *  # noqa: F401,F403
from src.feature_engineering.panel_builder import (  # type: ignore
    PanelConfig,
    temporal_train_val_test_split,
)

from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class HurdleTemporalTrainer:
    """
    Two-stage hurdle model for temporal crash prediction.

    Stage 1 (binary): HistGradientBoostingClassifier — predicts P(crash occurs)
    Stage 2 (count):  HistGradientBoostingRegressor(Poisson) — predicts
                      E[crash_count | crash occurs], trained only on positive windows.

    Combined inference:
        λ_overall = P(crash_occurs) × E[crash_count | crash_occurs]
    """

    def __init__(
        self,
        random_state: int = 42,
        panel_config: Optional[PanelConfig] = None,
        lambda_cap: Optional[float] = 50.0,
    ):
        self.random_state = random_state
        self.panel_config = panel_config or PanelConfig()
        self.lambda_cap = lambda_cap
        self.stage1: Optional[HistGradientBoostingClassifier] = None
        self.stage2: Optional[HistGradientBoostingRegressor] = None
        self.scaler = StandardScaler()
        self.feature_columns: list[str] = []
        self.calibrator: Optional[IsotonicRegression] = None

    def _prepare_features(
        self,
        panel: pd.DataFrame,
        target_col: str = "future_crash_count",
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Extract feature matrix and target, mirroring TemporalCountModelTrainer."""
        exclude = {
            "segment_id", "FROM_INTERSECTION_ID", "TO_INTERSECTION_ID",
            "segment_centroid_lat", "segment_centroid_lon",
            "window_start", "future_window_start", "datetime_hour",
            "lat_grid", "lon_grid",
            "ROAD_CLASS", "season",
            "hour_of_day", "day_of_week", "month",
            "crash_count", "future_crash_count", "is_ksi", "fatalities",
            "sample_weight", "sample_weight_tail",
        }
        exclude.update({c for c in panel.columns if c.startswith("sample_weight")})

        feature_cols = [c for c in panel.columns if c not in exclude]
        X = panel[feature_cols].copy().fillna(0)
        for col in X.columns:
            if X[col].dtype == "object":
                X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)
        X = X.astype(float)
        y = panel[target_col].astype(float).copy()
        return X, y

    def train_temporal_count_model(
        self,
        panel: pd.DataFrame,
        target_col: str = "future_crash_count",
        sample_weight_col: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Train the two-stage hurdle model with temporal train/val/test split.

        Returns a results dict compatible with TemporalCountModelTrainer.
        """
        train_data, val_data, test_data = temporal_train_val_test_split(panel)

        X_train, y_train = self._prepare_features(train_data, target_col)
        X_val, y_val = self._prepare_features(val_data, target_col)
        X_test, y_test = self._prepare_features(test_data, target_col)

        self.feature_columns = list(X_train.columns)

        X_train_s = self.scaler.fit_transform(X_train)
        X_val_s = self.scaler.transform(X_val)
        X_test_s = self.scaler.transform(X_test)

        sample_weight_train = None
        if sample_weight_col is not None and sample_weight_col in train_data.columns:
            sample_weight_train = train_data[sample_weight_col].astype(float).values

        # --- Stage 1: binary crash/no-crash classifier ---
        y_binary_train = (y_train > 0).astype(int)
        self.stage1 = HistGradientBoostingClassifier(
            loss="log_loss",
            max_depth=6,
            learning_rate=0.1,
            max_iter=300,
            random_state=self.random_state,
        )
        self.stage1.fit(X_train_s, y_binary_train, sample_weight=sample_weight_train)

        p_val_raw = self.stage1.predict_proba(X_val_s)[:, 1]
        y_val_binary = (y_val > 0).astype(int)
        self.calibrator = IsotonicRegression(out_of_bounds="clip")
        self.calibrator.fit(p_val_raw, y_val_binary)

        logger.info(
            "Stage 1 training: %d total, %d positive (%.1f%%)",
            len(y_binary_train),
            int(y_binary_train.sum()),
            100.0 * y_binary_train.mean(),
        )

        # --- Stage 2: count regressor on positive windows only ---
        pos_mask = y_train > 0
        n_pos = int(pos_mask.sum())
        logger.info("Stage 2 training: %d positive windows (of %d total)", n_pos, len(y_train))
        if n_pos == 0:
            raise ValueError(
                "No positive crash windows in training set — Stage 2 cannot be trained. "
                "Check panel construction and target column."
            )

        w_pos = sample_weight_train[pos_mask] if sample_weight_train is not None else None
        self.stage2 = HistGradientBoostingRegressor(
            loss="poisson",
            max_depth=6,
            learning_rate=0.1,
            max_iter=300,
            random_state=self.random_state,
        )
        self.stage2.fit(X_train_s[pos_mask], y_train[pos_mask], sample_weight=w_pos)

        # --- Evaluate on test set ---
        y_pred = self.predict_lambda(pd.DataFrame(X_test, columns=self.feature_columns))
        if self.lambda_cap is not None:
            y_pred = np.clip(y_pred, 0.0, self.lambda_cap)

        mae = float(np.mean(np.abs(y_pred - y_test)))
        rmse = float(np.sqrt(np.mean((y_pred - y_test) ** 2)))
        eps = 1e-9
        yt = np.maximum(np.asarray(y_test), eps)
        yp = np.maximum(y_pred, eps)
        poisson_deviance = float(2 * np.mean(yp - yt + yt * np.log(yt / yp)))

        test_data_with_pred = test_data.copy()
        test_data_with_pred["y_pred"] = y_pred

        logger.info(
            "Hurdle model evaluation — MAE=%.4f, RMSE=%.4f, Poisson dev=%.4f",
            mae, rmse, poisson_deviance,
        )

        return {
            "mae": mae,
            "rmse": rmse,
            "poisson_deviance": poisson_deviance,
            "y_test": y_test,
            "y_pred": y_pred,
            "mean_train_y": float(np.mean(y_train)),
            "test_data_with_pred": test_data_with_pred,
            "X_test": X_test,
            "feature_columns": self.feature_columns,
            "calibration": {"fitted": True},
        }

    def predict_lambda(self, X: pd.DataFrame) -> np.ndarray:
        """Combined hurdle prediction: λ = P(crash) × E[count | crash]."""
        if self.stage1 is None or self.stage2 is None:
            raise ValueError("Model not trained. Call train_temporal_count_model() first.")
        X_feat = X[self.feature_columns].fillna(0)
        for col in X_feat.columns:
            if X_feat[col].dtype == "object":
                X_feat[col] = pd.to_numeric(X_feat[col], errors="coerce").fillna(0)
        X_feat = X_feat.astype(float)
        X_scaled = self.scaler.transform(X_feat)
        p_crash = self.stage1.predict_proba(X_scaled)[:, 1]
        lambda_given_crash = np.clip(self.stage2.predict(X_scaled), 0.0, None)
        pred = p_crash * lambda_given_crash
        if self.lambda_cap is not None:
            pred = np.clip(pred, 0.0, self.lambda_cap)
        return pred

    def lambda_to_window_probability(self, lambda_window: np.ndarray) -> np.ndarray:
        """Not implemented — use predict_crash_probability() for Stage 1 probabilities."""
        raise NotImplementedError(
            "Use predict_crash_probability() for direct Stage 1 probabilities."
        )

    def predict_crash_probability(self, X: pd.DataFrame) -> np.ndarray:
        """Return calibrated Stage 1 P(crash) for a feature matrix."""
        if self.stage1 is None:
            raise ValueError("Model not trained.")
        X_feat = X[self.feature_columns].fillna(0).astype(float)
        X_scaled = self.scaler.transform(X_feat)
        p_raw = self.stage1.predict_proba(X_scaled)[:, 1]
        if self.calibrator is not None:
            return self.calibrator.transform(p_raw)
        return p_raw

    def save_model(self, filepath: str) -> None:
        model_data = {
            "stage1": self.stage1,
            "stage2": self.stage2,
            "scaler": self.scaler,
            "feature_columns": self.feature_columns,
            "panel_config": self.panel_config,
            "calibrator": self.calibrator,
            "lambda_cap": self.lambda_cap,
        }
        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)
        logger.info("Hurdle model saved to %s", filepath)

    def load_model(self, filepath: str) -> None:
        with open(filepath, "rb") as f:
            model_data = pickle.load(f)
        self.stage1 = model_data["stage1"]
        self.stage2 = model_data["stage2"]
        self.scaler = model_data["scaler"]
        self.feature_columns = model_data["feature_columns"]
        self.panel_config = model_data.get("panel_config", PanelConfig())
        self.calibrator = model_data.get("calibrator", None)
        self.lambda_cap = model_data.get("lambda_cap", 50.0)
        logger.info("Hurdle model loaded from %s", filepath)
