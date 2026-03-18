"""
Compatibility shim — re-exports all model classes from their respective submodules.

Individual models live in:
  src/models/legacy_classifier/  — ModelTrainer (RandomForest)
  src/models/temporal_count/     — TemporalCountModelTrainer (HistGBT Poisson)
  src/models/hurdle/             — HurdleTemporalTrainer (two-stage hurdle)
"""

import logging
import pickle
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd

from .legacy_classifier.model_trainer import ModelTrainer
from .temporal_count.model_trainer import TemporalCountModelTrainer
from .hurdle.model_trainer import HurdleTemporalTrainer

warnings.filterwarnings("ignore")

# ML libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight

# Count-regression / temporal modeling
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.isotonic import IsotonicRegression

# Add parent directory to path for config import
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import *  # noqa: F401,F403
from src.feature_engineering.panel_builder import (  # type: ignore
    PanelConfig,
    temporal_train_val_test_split,
)

logger = logging.getLogger(__name__)

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline

    SMOTE_AVAILABLE = True
except (ImportError, AttributeError) as e:
    SMOTE_AVAILABLE = False
    logger.warning(
        f"SMOTE not available ({type(e).__name__}: {e}). Using class weights instead."
    )


class ModelTrainer:
    """
    Model trainer for crash risk prediction using Random Forest
    """

    def __init__(self, random_state: int = 42):
        """
        Initialize the model trainer

        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        # Always fit on all possible classes
        self.label_encoder.fit(["low", "medium", "high"])
        self.feature_columns = []
        self.class_weights = None
        self.best_params = None
        self.cv_scores = None

    def prepare_features(
        self, data: gpd.GeoDataFrame
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and labels for training

        Args:
            data: GeoDataFrame with features and risk labels

        Returns:
            Tuple of (features, labels)
        """
        logger.info("Preparing features for model training...")

        # Define feature columns (exclude non-feature columns and target-encoding features)
        exclude_columns = [
            "geometry",
            "segment_id",
            "_id",
            "CENTRELINE_ID",
            "LINEAR_NAME_ID",
            "LINEAR_NAME_FULL",
            "LINEAR_NAME_FULL_LEGAL",
            "ADDRESS_L",
            "ADDRESS_R",
            "PARITY_L",
            "PARITY_R",
            "LO_NUM_L",
            "HI_NUM_L",
            "LO_NUM_R",
            "HI_NUM_R",
            "BEGIN_ADDR_POINT_ID_L",
            "END_ADDR_POINT_ID_L",
            "BEGIN_ADDR_POINT_ID_R",
            "END_ADDR_POINT_ID_R",
            "BEGIN_ADDR_L",
            "END_ADDR_L",
            "BEGIN_ADDR_R",
            "END_ADDR_R",
            "LOW_NUM_ODD",
            "HIGH_NUM_ODD",
            "LOW_NUM_EVEN",
            "HIGH_NUM_EVEN",
            "LINEAR_NAME",
            "LINEAR_NAME_TYPE",
            "LINEAR_NAME_DIR",
            "LINEAR_NAME_DESC",
            "LINEAR_NAME_LABEL",
            "FROM_INTERSECTION_ID",
            "TO_INTERSECTION_ID",
            "ONEWAY_DIR_CODE",
            "ONEWAY_DIR_CODE_DESC",
            "FEATURE_CODE",
            "FEATURE_CODE_DESC",
            "JURISDICTION",
            "CENTRELINE_STATUS",
            "OBJECTID",
            "MI_PRINX",
            # Target-encoding features causing data leakage
            "risk_label",
            "risk_level",
            "fatality_flag",
            "risk_score_raw",
            "severity_index",
            # Direct outcome variables used for labeling (data leakage)
            "num_total_crashes",
            "num_ksi_crashes",
            "fatality_count",
            "has_crashes",
            "has_ksi",
            "has_fatalities",
            "ksi_ratio",
            "fatality_ratio",
            "crash_density",
            "ksi_density",
            "length_crash_interaction",
            "length_ksi_interaction",
        ]

        # Get feature columns
        feature_columns = [col for col in data.columns if col not in exclude_columns]
        self.feature_columns = feature_columns

        # Extract features and labels
        X = data[feature_columns].copy()
        y = data["risk_label"].copy()

        # Handle missing values
        X = X.fillna(0)

        # Convert categorical columns to numeric
        for col in X.columns:
            if X[col].dtype == "object":
                # Convert to numeric, coercing errors to NaN, then fill with 0
                X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)

        # Ensure all data is numeric
        X = X.astype(float)

        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)

        logger.info(f"Prepared {len(feature_columns)} features for {len(X)} samples")
        logger.info(
            f"Label distribution: {dict(zip(self.label_encoder.classes_, np.bincount(y_encoded)))}"
        )

        return X, y_encoded

    def handle_class_imbalance(
        self, X: pd.DataFrame, y: np.ndarray
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Handle class imbalance using SMOTE or class weights

        Args:
            X: Feature matrix
            y: Target labels

        Returns:
            Balanced feature matrix and labels
        """
        if SMOTE_AVAILABLE:
            logger.info("Handling class imbalance with SMOTE...")

            # Apply SMOTE for oversampling minority classes
            smote = SMOTE(random_state=self.random_state, k_neighbors=3)
            X_balanced, y_balanced = smote.fit_resample(X, y)

            logger.info(f"Original class distribution: {np.bincount(y)}")
            logger.info(f"Balanced class distribution: {np.bincount(y_balanced)}")

            return X_balanced, y_balanced
        else:
            logger.info(
                "SMOTE not available. Using original data with class weights..."
            )
            return X, y

    def train_model(
        self, X: pd.DataFrame, y: np.ndarray, use_hyperparameter_tuning: bool = True
    ) -> Dict[str, Any]:
        """
        Train the Random Forest model

        Args:
            X: Feature matrix
            y: Target labels
            use_hyperparameter_tuning: Whether to use GridSearchCV

        Returns:
            Dictionary with training results
        """
        logger.info("Training Random Forest model...")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        if use_hyperparameter_tuning:
            # Define parameter grid for GridSearchCV
            param_grid = {
                "n_estimators": [100, 200, 300],
                "max_depth": [10, 15, 20, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "class_weight": ["balanced", "balanced_subsample"],
            }

            # Initialize base model
            base_model = RandomForestClassifier(random_state=self.random_state)

            # Perform GridSearchCV
            grid_search = GridSearchCV(
                base_model, param_grid, cv=5, scoring="f1_macro", n_jobs=-1, verbose=1
            )

            grid_search.fit(X_train_scaled, y_train)

            # Get best model
            self.model = grid_search.best_estimator_
            self.best_params = grid_search.best_params_

            logger.info(f"Best parameters: {self.best_params}")
            logger.info(f"Best CV score: {grid_search.best_score_:.4f}")

        else:
            # Train with default parameters
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=self.random_state,
            )

            self.model.fit(X_train_scaled, y_train)

        # Make predictions
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)

        # Calculate basic metrics
        accuracy = accuracy_score(y_test, y_pred)

        # Cross-validation scores
        cv_scores = cross_val_score(
            self.model, X_train_scaled, y_train, cv=5, scoring="f1_macro"
        )
        self.cv_scores = cv_scores

        # Calculate detailed performance metrics
        from sklearn.metrics import (
            classification_report,
            confusion_matrix,
            precision_recall_fscore_support,
        )

        # Classification report
        class_report = classification_report(y_test, y_pred, output_dict=True)

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_test, y_pred, average=None
        )

        # Prediction confidence analysis
        max_proba = np.max(y_pred_proba, axis=1)
        confidence_analysis = {
            "mean_confidence": np.mean(max_proba),
            "confidence_when_correct": np.mean(max_proba[y_test == y_pred]),
            "confidence_when_wrong": np.mean(max_proba[y_test != y_pred]),
            "high_confidence_errors": np.sum((max_proba > 0.8) & (y_test != y_pred)),
        }

        # Prepare results
        results = {
            "accuracy": accuracy,
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "best_params": self.best_params,
            "feature_importance": dict(
                zip(self.feature_columns, self.model.feature_importances_)
            ),
            "y_test": y_test,
            "y_pred": y_pred,
            "y_pred_proba": y_pred_proba,
            "X_test": X_test,
            "X_test_scaled": X_test_scaled,
            "classification_report": class_report,
            "confusion_matrix": cm,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
            "confidence_analysis": confidence_analysis,
        }

        logger.info(f"Model training completed!")
        logger.info(f"Test accuracy: {accuracy:.4f}")
        logger.info(
            f"CV F1-score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})"
        )

        return results

    def save_model(self, filepath: str):
        """
        Save the trained model

        Args:
            filepath: Path to save the model
        """
        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "label_encoder": self.label_encoder,
            "feature_columns": self.feature_columns,
            "best_params": self.best_params,
            "cv_scores": self.cv_scores,
        }

        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)

        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """
        Load a trained model

        Args:
            filepath: Path to the saved model
        """
        with open(filepath, "rb") as f:
            model_data = pickle.load(f)

        self.model = model_data["model"]
        self.scaler = model_data["scaler"]
        self.label_encoder = model_data["label_encoder"]
        self.feature_columns = model_data["feature_columns"]
        self.best_params = model_data["best_params"]
        self.cv_scores = model_data["cv_scores"]

        logger.info(f"Model loaded from {filepath}")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data

        Args:
            X: Feature matrix

        Returns:
            Predicted labels
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")

        # Ensure same features
        X = X[self.feature_columns].fillna(0)

        # Scale features
        X_scaled = self.scaler.transform(X)

        # Make predictions
        predictions = self.model.predict(X_scaled)

        return predictions


class TemporalCountModelTrainer:
    """
    Trainer for count-based crash models on temporal panel data.

    This class is designed to work with the panel produced by
    `feature_engineering.panel_builder.build_panel_dataset` and trains
    a model that outputs an approximate crash rate λ (crashes per window,
    which can be converted to crashes/hour).
    """

    def __init__(
        self,
        random_state: int = 42,
        panel_config: Optional[PanelConfig] = None,
        lambda_cap: Optional[float] = 50.0,
    ):
        self.random_state = random_state
        self.panel_config = panel_config or PanelConfig()
        self.lambda_cap = (
            lambda_cap  # cap predicted λ (crashes per window) for stability and routing
        )
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
        """
        Prepare features and target from a temporal panel.

        Excludes:
        - identifiers and time columns
        - outcome columns that would leak if used as features
        """
        logger.info("Preparing panel features for temporal count model training...")

        if target_col not in panel.columns:
            raise ValueError(f"Target column '{target_col}' not found in panel.")

        exclude = {
            # Identifiers — these let the model memorise segments, not learn patterns
            "segment_id",
            "FROM_INTERSECTION_ID",
            "TO_INTERSECTION_ID",
            # Spatial coordinates used only for weather joins, not as features
            "segment_centroid_lat",
            "segment_centroid_lon",
            # Time columns
            "window_start",
            "future_window_start",
            "datetime_hour",
            "lat_grid",
            "lon_grid",
            # Raw categoricals replaced by one-hot / integer encodings
            "ROAD_CLASS",
            "season",
            # Raw integer hour/day kept only for cyclical encoding
            "hour_of_day",
            "day_of_week",
            "month",
            # Outcome variables (do not use as predictors)
            "crash_count",
            "future_crash_count",
            "is_ksi",
            "fatalities",
            # Training-only metadata
            "sample_weight",
            "sample_weight_tail",
        }

        # Exclude any additional sample-weight style columns to avoid leakage.
        exclude.update({c for c in panel.columns if c.startswith("sample_weight")})

        if explicit_feature_cols is not None:
            feature_cols = [c for c in explicit_feature_cols if c not in exclude]
        else:
            feature_cols = [c for c in panel.columns if c not in exclude]

        X = panel[feature_cols].copy()
        y = panel[target_col].astype(float).copy()

        # Handle missing values and ensure numeric
        X = X.fillna(0)
        for col in X.columns:
            if X[col].dtype == "object":
                X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)
        X = X.astype(float)

        self.feature_columns = feature_cols
        logger.info(
            "Panel features prepared: %d features, %d samples.",
            len(feature_cols),
            len(X),
        )
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
        Train a temporal count model using ordered window_start splits.

        - Uses temporal_train_val_test_split for realistic evaluation
        - Trains HistGradientBoostingRegressor with Poisson loss to approximate λ
        """
        # Temporal split
        train_data, val_data, test_data = temporal_train_val_test_split(panel)

        # Prepare features/targets
        X_train, y_train = self.prepare_panel_features(
            train_data, target_col=target_col
        )
        X_val, y_val = self.prepare_panel_features(val_data, target_col=target_col)
        X_test, y_test = self.prepare_panel_features(test_data, target_col=target_col)

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)

        # Optional sample weights (e.g. for sampled zero windows)
        sample_weight_train = None
        if sample_weight_col is not None and sample_weight_col in train_data.columns:
            sample_weight_train = train_data[sample_weight_col].astype(float).values

        if use_hyperparameter_tuning:
            # Simple manual search over a few key parameters
            candidates = [
                {"max_depth": 5, "learning_rate": 0.1, "max_iter": 300},
                {"max_depth": 7, "learning_rate": 0.05, "max_iter": 400},
            ]
            best_cfg = None
            best_score = -np.inf
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
                # Negative RMSE as score (we want to minimize RMSE)
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

        # Evaluate on test set (apply prediction cap for stability)
        y_pred = self.model.predict(X_test_scaled)  # type: ignore[arg-type]
        y_pred = np.clip(y_pred, 0.0, None)
        if self.lambda_cap is not None:
            y_pred = np.clip(y_pred, 0.0, self.lambda_cap)
        mae = np.mean(np.abs(y_pred - y_test))
        rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))

        # Very simple Poisson deviance approximation (guard against zeros)
        eps = 1e-9
        y_true_clipped = np.maximum(y_test, eps)
        y_pred_clipped = np.maximum(y_pred, eps)
        poisson_deviance = 2 * np.mean(
            y_pred_clipped
            - y_true_clipped
            + y_true_clipped * np.log(y_true_clipped / y_pred_clipped)
        )

        # Calibration on validation set (probability that a window has ≥1 crash)
        lambda_val = self.model.predict(X_val_scaled)  # type: ignore[arg-type]
        lambda_val = np.clip(lambda_val, 0.0, None)
        P_val_raw = 1.0 - np.exp(-lambda_val)  # probability of ≥1 crash in the window
        y_val_binary = (y_val > 0).astype(int)

        self.calibrator = IsotonicRegression(out_of_bounds="clip")
        self.calibrator.fit(P_val_raw, y_val_binary)

        # Attach test set with predictions for outlier inspection
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
            "calibration": {
                "fitted": True,
            },
        }

        logger.info(
            "Temporal count model evaluation — MAE=%.4f, RMSE=%.4f, Poisson dev=%.4f",
            mae,
            rmse,
            poisson_deviance,
        )

        return results

    def save_model(self, filepath: str) -> None:
        """Persist temporal count model and metadata."""
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
        """Load temporal count model."""
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
        """
        Predict crash counts per window (λ_window) for new data.

        Predictions are clipped to [0, lambda_cap] when lambda_cap is set.
        Caller is responsible for converting to crashes/hour or to traversal
        probability (e.g., λ_hour = λ_window / window_size_hours).
        """
        if self.model is None:
            raise ValueError(
                "Model not trained. Call train_temporal_count_model() first."
            )

        X = X[self.feature_columns].fillna(0)
        for col in X.columns:
            if X[col].dtype == "object":
                X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)
        X = X.astype(float)
        X_scaled = self.scaler.transform(X)
        pred = self.model.predict(X_scaled)  # type: ignore[arg-type]
        pred = np.clip(pred, 0.0, None)
        if self.lambda_cap is not None:
            pred = np.clip(pred, 0.0, self.lambda_cap)
        return pred

    def lambda_to_window_probability(self, lambda_window: np.ndarray) -> np.ndarray:
        """
        Convert λ_window (expected crashes per window) to probability of ≥1 crash
        in the window, applying isotonic calibration if available.

        P_raw = 1 - exp(-λ_window)
        """
        lambda_window = np.clip(lambda_window, 0.0, None)
        P_raw = 1.0 - np.exp(-lambda_window)
        if self.calibrator is not None:
            return self.calibrator.transform(P_raw)
        return P_raw


class HurdleTemporalTrainer:
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

    def prepare_panel_features(
        self,
        panel: pd.DataFrame,
        target_col: str = "future_crash_count",
        explicit_feature_cols: Optional[list[str]] = None,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Extract feature matrix and target; same API as TemporalCountModelTrainer (explicit_feature_cols unused)."""
        exclude = {
            "segment_id",
            "FROM_INTERSECTION_ID",
            "TO_INTERSECTION_ID",
            "segment_centroid_lat",
            "segment_centroid_lon",
            "window_start",
            "future_window_start",
            "datetime_hour",
            "lat_grid",
            "lon_grid",
            "ROAD_CLASS",
            "season",
            "hour_of_day",
            "day_of_week",
            "month",
            "crash_count",
            "future_crash_count",
            "is_ksi",
            "fatalities",
            "sample_weight",
            "sample_weight_tail",
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

        X_train, y_train = self.prepare_panel_features(
            train_data, target_col=target_col
        )
        X_val, y_val = self.prepare_panel_features(val_data, target_col=target_col)
        X_test, y_test = self.prepare_panel_features(test_data, target_col=target_col)

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

        # Calibrate Stage 1 on validation set
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
        logger.info(
            "Stage 2 training: %d positive windows (of %d total)", n_pos, len(y_train)
        )
        if n_pos == 0:
            raise ValueError(
                "No positive crash windows in training set — Stage 2 cannot be trained. "
                "Check panel construction and target column."
            )

        w_pos = (
            sample_weight_train[pos_mask] if sample_weight_train is not None else None
        )
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
            mae,
            rmse,
            poisson_deviance,
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
        """
        Combined hurdle prediction: λ = P(crash) × E[count | crash].

        Applies lambda_cap when set.
        """
        if self.stage1 is None or self.stage2 is None:
            raise ValueError(
                "Model not trained. Call train_temporal_count_model() first."
            )

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
        """
        For the hurdle model, window probability is P(crash) from Stage 1.
        lambda_window is ignored; this method exists for interface compatibility.
        """
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
        """Persist hurdle model and metadata."""
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
        """Load hurdle model."""
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


def test_model_trainer():
    """
    Test function for model trainer
    """
    from data_processing.data_loader import load_and_clean_data
    from data_processing.spatial_join_fast import perform_spatial_join_fast
    from feature_engineering.feature_creator import create_segment_features
    from feature_engineering.label_generator import generate_risk_labels

    logging.basicConfig(level=logging.INFO)
    data_dir = Path("data")

    # Load and process data
    collision_data, ksi_data, road_network = load_and_clean_data(data_dir)
    segment_crashes = perform_spatial_join_fast(collision_data, ksi_data, road_network)
    segment_features = create_segment_features(segment_crashes, road_network)
    labeled_segments = generate_risk_labels(segment_features)

    # Initialize and train model
    trainer = ModelTrainer()
    X, y = trainer.prepare_features(labeled_segments)
    X_balanced, y_balanced = trainer.handle_class_imbalance(X, y)
    results = trainer.train_model(
        X_balanced, y_balanced, use_hyperparameter_tuning=False
    )

    # Print results
    print(f"\nModel Training Results:")
    print(f"Test Accuracy: {results['accuracy']:.4f}")
    print(f"CV F1-Score: {results['cv_mean']:.4f} (+/- {results['cv_std'] * 2:.4f})")
    print(f"Best Parameters: {results['best_params']}")

    # Show top 10 feature importance
    feature_importance = results["feature_importance"]
    top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[
        :10
    ]
    print(f"\nTop 10 Feature Importance:")
    for feature, importance in top_features:
        print(f"  {feature}: {importance:.4f}")

    return trainer, results


if __name__ == "__main__":
    test_model_trainer()
