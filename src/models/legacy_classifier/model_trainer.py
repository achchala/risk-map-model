"""
Legacy RandomForest classifier for Toronto Road Segment Crash Risk Prediction.

Predicts low/medium/high risk labels per road segment using static features.
"""

import pandas as pd
import numpy as np
import geopandas as gpd
from pathlib import Path
import logging
import pickle
import sys
from typing import Tuple, Dict, Any

import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight

sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from config import *  # noqa: F401,F403

logger = logging.getLogger(__name__)

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except (ImportError, AttributeError) as e:
    SMOTE_AVAILABLE = False
    logger.warning(f"SMOTE not available ({type(e).__name__}: {e}). Using class weights instead.")


class ModelTrainer:
    """
    Model trainer for crash risk prediction using Random Forest
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(['low', 'medium', 'high'])
        self.feature_columns = []
        self.class_weights = None
        self.best_params = None
        self.cv_scores = None

    def prepare_features(self, data: gpd.GeoDataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        logger.info("Preparing features for model training...")

        exclude_columns = [
            'geometry', 'segment_id', '_id', 'CENTRELINE_ID', 'LINEAR_NAME_ID',
            'LINEAR_NAME_FULL', 'LINEAR_NAME_FULL_LEGAL', 'ADDRESS_L', 'ADDRESS_R',
            'PARITY_L', 'PARITY_R', 'LO_NUM_L', 'HI_NUM_L', 'LO_NUM_R', 'HI_NUM_R',
            'BEGIN_ADDR_POINT_ID_L', 'END_ADDR_POINT_ID_L', 'BEGIN_ADDR_POINT_ID_R',
            'END_ADDR_POINT_ID_R', 'BEGIN_ADDR_L', 'END_ADDR_L', 'BEGIN_ADDR_R',
            'END_ADDR_R', 'LOW_NUM_ODD', 'HIGH_NUM_ODD', 'LOW_NUM_EVEN', 'HIGH_NUM_EVEN',
            'LINEAR_NAME', 'LINEAR_NAME_TYPE', 'LINEAR_NAME_DIR', 'LINEAR_NAME_DESC',
            'LINEAR_NAME_LABEL', 'FROM_INTERSECTION_ID', 'TO_INTERSECTION_ID',
            'ONEWAY_DIR_CODE', 'ONEWAY_DIR_CODE_DESC', 'FEATURE_CODE', 'FEATURE_CODE_DESC',
            'JURISDICTION', 'CENTRELINE_STATUS', 'OBJECTID', 'MI_PRINX',
            # Target-encoding features causing data leakage
            'risk_label', 'risk_level', 'fatality_flag', 'risk_score_raw', 'severity_index',
            # Direct outcome variables used for labeling (data leakage)
            'num_total_crashes', 'num_ksi_crashes', 'fatality_count', 'has_crashes', 'has_ksi', 'has_fatalities',
            'ksi_ratio', 'fatality_ratio', 'crash_density', 'ksi_density',
            'length_crash_interaction', 'length_ksi_interaction',
        ]

        feature_columns = [col for col in data.columns if col not in exclude_columns]
        self.feature_columns = feature_columns

        X = data[feature_columns].copy()
        y = data['risk_label'].copy()

        X = X.fillna(0)
        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
        X = X.astype(float)

        y_encoded = self.label_encoder.fit_transform(y)

        logger.info(f"Prepared {len(feature_columns)} features for {len(X)} samples")
        logger.info(f"Label distribution: {dict(zip(self.label_encoder.classes_, np.bincount(y_encoded)))}")

        return X, y_encoded

    def handle_class_imbalance(self, X: pd.DataFrame, y: np.ndarray) -> Tuple[pd.DataFrame, np.ndarray]:
        if SMOTE_AVAILABLE:
            logger.info("Handling class imbalance with SMOTE...")
            smote = SMOTE(random_state=self.random_state, k_neighbors=3)
            X_balanced, y_balanced = smote.fit_resample(X, y)
            logger.info(f"Original class distribution: {np.bincount(y)}")
            logger.info(f"Balanced class distribution: {np.bincount(y_balanced)}")
            return X_balanced, y_balanced
        else:
            logger.info("SMOTE not available. Using original data with class weights...")
            return X, y

    def train_model(self, X: pd.DataFrame, y: np.ndarray,
                    use_hyperparameter_tuning: bool = True) -> Dict[str, Any]:
        logger.info("Training Random Forest model...")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        if use_hyperparameter_tuning:
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'class_weight': ['balanced', 'balanced_subsample'],
            }
            base_model = RandomForestClassifier(random_state=self.random_state)
            grid_search = GridSearchCV(
                base_model, param_grid, cv=5, scoring='f1_macro',
                n_jobs=-1, verbose=1
            )
            grid_search.fit(X_train_scaled, y_train)
            self.model = grid_search.best_estimator_
            self.best_params = grid_search.best_params_
            logger.info(f"Best parameters: {self.best_params}")
            logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
        else:
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced',
                random_state=self.random_state,
            )
            self.model.fit(X_train_scaled, y_train)

        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)

        accuracy = accuracy_score(y_test, y_pred)
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5, scoring='f1_macro')
        self.cv_scores = cv_scores

        from sklearn.metrics import precision_recall_fscore_support
        class_report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average=None)

        max_proba = np.max(y_pred_proba, axis=1)
        confidence_analysis = {
            'mean_confidence': np.mean(max_proba),
            'confidence_when_correct': np.mean(max_proba[y_test == y_pred]),
            'confidence_when_wrong': np.mean(max_proba[y_test != y_pred]),
            'high_confidence_errors': np.sum((max_proba > 0.8) & (y_test != y_pred)),
        }

        results = {
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'best_params': self.best_params,
            'feature_importance': dict(zip(self.feature_columns, self.model.feature_importances_)),
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'X_test': X_test,
            'X_test_scaled': X_test_scaled,
            'classification_report': class_report,
            'confusion_matrix': cm,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': support,
            'confidence_analysis': confidence_analysis,
        }

        logger.info(f"Model training completed!")
        logger.info(f"Test accuracy: {accuracy:.4f}")
        logger.info(f"CV F1-score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

        return results

    def save_model(self, filepath: str) -> None:
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_columns': self.feature_columns,
            'best_params': self.best_params,
            'cv_scores': self.cv_scores,
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str) -> None:
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']
        self.feature_columns = model_data['feature_columns']
        self.best_params = model_data['best_params']
        self.cv_scores = model_data['cv_scores']
        logger.info(f"Model loaded from {filepath}")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        X = X[self.feature_columns].fillna(0)
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
