"""
Standalone temporal count model for crash rate prediction.

Trains a HistGradientBoostingRegressor with Poisson loss on temporal panel data,
outputting λ (expected crashes per window). Includes isotonic calibration for
converting λ to P(≥1 crash).
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

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class TemporalCountModelTrainer:
    """
    Trainer for count-based crash models on temporal panel data.

    Trains a model that outputs an approximate crash rate λ (crashes per window).
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
        self.model: Optional[HistGradientBoostingRegressor] = None
        self.scaler = StandardScaler()
        self.feature_columns: list[str] = []
        self.cv_scores: Optional[np.ndarray] = None
        self.calibrator: Optional[IsotonicRegression] = None

    def prepare_panel_features(
        self,
        panel: pd.DataFrame,
        target_col: str = "future_crash_count",
        explicit_feature_cols: Optional[list[str]] = None,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        logger.info("Preparing panel features for temporal count model training...")

        if target_col not in panel.columns:
            raise ValueError(f"Target column '{target_col}' not found in panel.")

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

        if explicit_feature_cols is not None:
            feature_cols = [c for c in explicit_feature_cols if c not in exclude]
        else:
            feature_cols = [c for c in panel.columns if c not in exclude]

        X = panel[feature_cols].copy()
        y = panel[target_col].astype(float).copy()

        X = X.fillna(0)
        for col in X.columns:
            if X[col].dtype == "object":
                X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)
        X = X.astype(float)

        self.feature_columns = feature_cols
        logger.info("Panel features prepared: %d features, %d samples.", len(feature_cols), len(X))
        logger.info("Feature columns: %s", feature_cols)

        return X, y

    def train_temporal_count_model(
        self,
        panel: pd.DataFrame,
        target_col: str = "future_crash_count",
        sample_weight_col: Optional[str] = None,
        use_hyperparameter_tuning: bool = False,
    ) -> Dict[str, Any]:
        """
        Train using ordered window_start splits (temporal train/val/test).
        """
        train_data, val_data, test_data = temporal_train_val_test_split(panel)

        X_train, y_train = self.prepare_panel_features(train_data, target_col=target_col)
        X_val, y_val = self.prepare_panel_features(val_data, target_col=target_col)
        X_test, y_test = self.prepare_panel_features(test_data, target_col=target_col)

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)

        sample_weight_train = None
        if sample_weight_col is not None and sample_weight_col in train_data.columns:
            sample_weight_train = train_data[sample_weight_col].astype(float).values

        if use_hyperparameter_tuning:
            candidates = [
                {"max_depth": 5, "learning_rate": 0.1, "max_iter": 300},
                {"max_depth": 7, "learning_rate": 0.05, "max_iter": 400},
            ]
            best_score = -np.inf
            best_cfg = None
            for cfg in candidates:
                model = HistGradientBoostingRegressor(
                    loss="poisson",
                    max_depth=cfg["max_depth"],
                    learning_rate=cfg["learning_rate"],
                    max_iter=cfg["max_iter"],
                    random_state=self.random_state,
                )
                model.fit(X_train_scaled, y_train, sample_weight=sample_weight_train)
                val_pred = model.predict(X_val_scaled)
                rmse = np.sqrt(np.mean((val_pred - y_val) ** 2))
                score = -rmse
                logger.info("Config %s → val RMSE=%.4f", cfg, rmse)
                if score > best_score:
                    best_score = score
                    best_cfg = cfg
                    self.model = model
            logger.info("Best temporal count model config: %s", best_cfg)
        else:
            self.model = HistGradientBoostingRegressor(
                loss="poisson",
                max_depth=6,
                learning_rate=0.1,
                max_iter=300,
                random_state=self.random_state,
            )
            self.model.fit(X_train_scaled, y_train, sample_weight=sample_weight_train)

        y_pred = self.model.predict(X_test_scaled)
        y_pred = np.clip(y_pred, 0.0, None)
        if self.lambda_cap is not None:
            y_pred = np.clip(y_pred, 0.0, self.lambda_cap)

        mae = np.mean(np.abs(y_pred - y_test))
        rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))

        eps = 1e-9
        y_true_clipped = np.maximum(y_test, eps)
        y_pred_clipped = np.maximum(y_pred, eps)
        poisson_deviance = 2 * np.mean(
            y_pred_clipped - y_true_clipped + y_true_clipped * np.log(y_true_clipped / y_pred_clipped)
        )

        lambda_val = self.model.predict(X_val_scaled)
        lambda_val = np.clip(lambda_val, 0.0, None)
        P_val_raw = 1.0 - np.exp(-lambda_val)
        y_val_binary = (y_val > 0).astype(int)

        self.calibrator = IsotonicRegression(out_of_bounds="clip")
        self.calibrator.fit(P_val_raw, y_val_binary)

        test_data_with_pred = test_data.copy()
        test_data_with_pred["y_pred"] = y_pred

        results = {
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

        logger.info(
            "Temporal count model evaluation — MAE=%.4f, RMSE=%.4f, Poisson dev=%.4f",
            mae, rmse, poisson_deviance,
        )

        return results

    def save_model(self, filepath: str) -> None:
        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "feature_columns": self.feature_columns,
            "panel_config": self.panel_config,
            "calibrator": self.calibrator,
            "lambda_cap": self.lambda_cap,
        }
        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)
        logger.info("Temporal count model saved to %s", filepath)

    def load_model(self, filepath: str) -> None:
        with open(filepath, "rb") as f:
            model_data = pickle.load(f)
        self.model = model_data["model"]
        self.scaler = model_data["scaler"]
        self.feature_columns = model_data["feature_columns"]
        self.panel_config = model_data.get("panel_config", PanelConfig())
        self.calibrator = model_data.get("calibrator", None)
        self.lambda_cap = model_data.get("lambda_cap", 50.0)
        logger.info("Temporal count model loaded from %s", filepath)

    def predict_lambda(self, X: pd.DataFrame) -> np.ndarray:
        """Predict crash counts per window (λ_window), clipped to lambda_cap."""
        if self.model is None:
            raise ValueError("Model not trained. Call train_temporal_count_model() first.")
        X = X[self.feature_columns].fillna(0)
        for col in X.columns:
            if X[col].dtype == "object":
                X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)
        X = X.astype(float)
        X_scaled = self.scaler.transform(X)
        pred = self.model.predict(X_scaled)
        pred = np.clip(pred, 0.0, None)
        if self.lambda_cap is not None:
            pred = np.clip(pred, 0.0, self.lambda_cap)
        return pred

    def lambda_to_window_probability(self, lambda_window: np.ndarray) -> np.ndarray:
        """Convert λ_window to P(≥1 crash), with isotonic calibration if available."""
        lambda_window = np.clip(lambda_window, 0.0, None)
        P_raw = 1.0 - np.exp(-lambda_window)
        if self.calibrator is not None:
            return self.calibrator.transform(P_raw)
        return P_raw
