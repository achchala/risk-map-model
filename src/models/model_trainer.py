"""
Model Trainer for Toronto Road Segment Crash Risk Prediction

This module handles model training, hyperparameter tuning, and class imbalance.
"""

import pandas as pd
import numpy as np
import geopandas as gpd
from pathlib import Path
import logging
import pickle
import sys
from typing import Tuple, Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# ML libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    logger.warning("SMOTE not available. Using class weights instead.")

# Add parent directory to path for config import
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import *

logger = logging.getLogger(__name__)

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
        self.feature_columns = None
        self.class_weights = None
        self.best_params = None
        self.cv_scores = None
        
    def prepare_features(self, data: gpd.GeoDataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and labels for training
        
        Args:
            data: GeoDataFrame with features and risk labels
            
        Returns:
            Tuple of (features, labels)
        """
        logger.info("Preparing features for model training...")
        
        # Define feature columns (exclude non-feature columns)
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
            'JURISDICTION', 'CENTRELINE_STATUS', 'OBJECTID', 'MI_PRINX'
        ]
        
        # Get feature columns
        feature_columns = [col for col in data.columns if col not in exclude_columns]
        self.feature_columns = feature_columns
        
        # Extract features and labels
        X = data[feature_columns].copy()
        y = data['risk_label'].copy()
        
        # Handle missing values
        X = X.fillna(0)
        
        # Convert categorical columns to numeric
        for col in X.columns:
            if X[col].dtype == 'object':
                # Convert to numeric, coercing errors to NaN, then fill with 0
                X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
        
        # Ensure all data is numeric
        X = X.astype(float)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        logger.info(f"Prepared {len(feature_columns)} features for {len(X)} samples")
        logger.info(f"Label distribution: {dict(zip(self.label_encoder.classes_, np.bincount(y_encoded)))}")
        
        return X, y_encoded
    
    def handle_class_imbalance(self, X: pd.DataFrame, y: np.ndarray) -> Tuple[pd.DataFrame, np.ndarray]:
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
            logger.info("SMOTE not available. Using original data with class weights...")
            return X, y
    
    def train_model(self, X: pd.DataFrame, y: np.ndarray, 
                   use_hyperparameter_tuning: bool = True) -> Dict[str, Any]:
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
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'class_weight': ['balanced', 'balanced_subsample']
            }
            
            # Initialize base model
            base_model = RandomForestClassifier(random_state=self.random_state)
            
            # Perform GridSearchCV
            grid_search = GridSearchCV(
                base_model, param_grid, cv=5, scoring='f1_macro', 
                n_jobs=-1, verbose=1
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
                class_weight='balanced',
                random_state=self.random_state
            )
            
            self.model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = self.model.predict(X_test_scaled)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # Cross-validation scores
        cv_scores = cross_val_score(
            self.model, X_train_scaled, y_train, cv=5, scoring='f1_macro'
        )
        self.cv_scores = cv_scores
        
        # Prepare results
        results = {
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'best_params': self.best_params,
            'feature_importance': dict(zip(self.feature_columns, self.model.feature_importances_)),
            'y_test': y_test,
            'y_pred': y_pred,
            'X_test': X_test,
            'X_test_scaled': X_test_scaled
        }
        
        logger.info(f"Model training completed!")
        logger.info(f"Test accuracy: {accuracy:.4f}")
        logger.info(f"CV F1-score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return results
    
    def save_model(self, filepath: str):
        """
        Save the trained model
        
        Args:
            filepath: Path to save the model
        """
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_columns': self.feature_columns,
            'best_params': self.best_params,
            'cv_scores': self.cv_scores
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load a trained model
        
        Args:
            filepath: Path to the saved model
        """
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
    results = trainer.train_model(X_balanced, y_balanced, use_hyperparameter_tuning=False)
    
    # Print results
    print(f"\nModel Training Results:")
    print(f"Test Accuracy: {results['accuracy']:.4f}")
    print(f"CV F1-Score: {results['cv_mean']:.4f} (+/- {results['cv_std'] * 2:.4f})")
    print(f"Best Parameters: {results['best_params']}")
    
    # Show top 10 feature importance
    feature_importance = results['feature_importance']
    top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
    print(f"\nTop 10 Feature Importance:")
    for feature, importance in top_features:
        print(f"  {feature}: {importance:.4f}")
    
    return trainer, results

if __name__ == "__main__":
    test_model_trainer() 