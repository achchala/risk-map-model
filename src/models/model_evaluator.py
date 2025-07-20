"""
Model Evaluator for Toronto Road Segment Crash Risk Prediction

This module handles model evaluation, performance metrics, and result analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import sys
from typing import Dict, Any, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# ML libraries
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score,
    precision_recall_curve, roc_curve
)
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path for config import
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import *

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """
    Model evaluator for comprehensive performance analysis
    """
    
    def __init__(self):
        """Initialize the model evaluator"""
        self.results = {}
        self.class_names = ['low', 'medium', 'high']
        
    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray, 
                      y_proba: Optional[np.ndarray] = None,
                      feature_importance: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Comprehensive model evaluation
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (optional)
            feature_importance: Feature importance dictionary (optional)
            
        Returns:
            Dictionary with evaluation results
        """
        logger.info("Evaluating model performance...")
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        # Per-class metrics
        precision_per_class = precision_score(y_true, y_pred, average=None)
        recall_per_class = recall_score(y_true, y_pred, average=None)
        f1_per_class = f1_score(y_true, y_pred, average=None)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Classification report
        class_report = classification_report(y_true, y_pred, target_names=self.class_names, output_dict=True)
        
        # Store results
        self.results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'precision_per_class': precision_per_class,
            'recall_per_class': recall_per_class,
            'f1_per_class': f1_per_class,
            'confusion_matrix': cm,
            'classification_report': class_report,
            'y_true': y_true,
            'y_pred': y_pred,
            'y_proba': y_proba,
            'feature_importance': feature_importance
        }
        
        # Log results
        logger.info(f"Model Evaluation Results:")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall: {recall:.4f}")
        logger.info(f"  F1-Score: {f1:.4f}")
        
        return self.results
    
    def print_detailed_report(self):
        """Print detailed evaluation report"""
        if not self.results:
            logger.warning("No evaluation results available. Run evaluate_model() first.")
            return
        
        print("\n" + "="*60)
        print("MODEL EVALUATION REPORT")
        print("="*60)
        
        # Overall metrics
        print(f"\n OVERALL PERFORMANCE:")
        print(f"  Accuracy:  {self.results['accuracy']:.4f}")
        print(f"  Precision: {self.results['precision']:.4f}")
        print(f"  Recall:    {self.results['recall']:.4f}")
        print(f"  F1-Score:  {self.results['f1_score']:.4f}")
        
        # Per-class performance
        print(f"\n PER-CLASS PERFORMANCE:")
        for i, class_name in enumerate(self.class_names):
            print(f"  {class_name.upper()} RISK:")
            print(f"    Precision: {self.results['precision_per_class'][i]:.4f}")
            print(f"    Recall:    {self.results['recall_per_class'][i]:.4f}")
            print(f"    F1-Score:  {self.results['f1_per_class'][i]:.4f}")
        
        # Confusion matrix
        print(f"\n CONFUSION MATRIX:")
        cm = self.results['confusion_matrix']
        print("           Predicted")
        print("           Low  Med  High")
        print(f"Actual Low  {cm[0][0]:4d} {cm[0][1]:4d} {cm[0][2]:4d}")
        print(f"      Med   {cm[1][0]:4d} {cm[1][1]:4d} {cm[1][2]:4d}")
        print(f"      High  {cm[2][0]:4d} {cm[2][1]:4d} {cm[2][2]:4d}")
        
        # Classification report
        print(f"\n DETAILED CLASSIFICATION REPORT:")
        report = self.results['classification_report']
        for class_name in self.class_names:
            if class_name in report:
                metrics = report[class_name]
                print(f"  {class_name.upper()} RISK:")
                print(f"    Precision: {metrics['precision']:.4f}")
                print(f"    Recall:    {metrics['recall']:.4f}")
                print(f"    F1-Score:  {metrics['f1-score']:.4f}")
                print(f"    Support:   {metrics['support']}")
    
    def plot_confusion_matrix(self, save_path: Optional[str] = None):
        """Plot confusion matrix heatmap"""
        if not self.results:
            logger.warning("No evaluation results available. Run evaluate_model() first.")
            return
        
        plt.figure(figsize=(8, 6))
        cm = self.results['confusion_matrix']
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Risk Level')
        plt.ylabel('Actual Risk Level')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
        
        plt.show()
    
    def plot_feature_importance(self, top_n: int = 15, save_path: Optional[str] = None):
        """Plot feature importance"""
        if not self.results or not self.results['feature_importance']:
            logger.warning("No feature importance data available.")
            return
        
        feature_importance = self.results['feature_importance']
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        features, importance = zip(*top_features)
        
        plt.figure(figsize=(10, 8))
        bars = plt.barh(range(len(features)), importance)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_n} Feature Importance')
        plt.gca().invert_yaxis()
        
        # Color bars by importance
        colors = plt.cm.Reds(np.linspace(0.3, 0.8, len(bars)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {save_path}")
        
        plt.show()
    
    def analyze_misclassifications(self, X_test: pd.DataFrame) -> pd.DataFrame:
        """Analyze misclassified samples"""
        if not self.results:
            logger.warning("No evaluation results available. Run evaluate_model() first.")
            return pd.DataFrame()
        
        y_true = self.results['y_true']
        y_pred = self.results['y_pred']
        
        # Find misclassified samples
        misclassified_mask = y_true != y_pred
        misclassified_indices = np.where(misclassified_mask)[0]
        
        if len(misclassified_indices) == 0:
            logger.info("No misclassifications found!")
            return pd.DataFrame()
        
        # Create analysis DataFrame
        analysis_data = []
        for idx in misclassified_indices:
            true_label = self.class_names[y_true[idx]]
            pred_label = self.class_names[y_pred[idx]]
            
            # Get feature values for this sample
            sample_features = X_test.iloc[idx]
            
            analysis_data.append({
                'index': idx,
                'true_label': true_label,
                'predicted_label': pred_label,
                'error_type': f"{true_label} → {pred_label}",
                **sample_features.to_dict()
            })
        
        misclassified_df = pd.DataFrame(analysis_data)
        
        logger.info(f"Found {len(misclassified_indices)} misclassified samples")
        logger.info(f"Error distribution: {misclassified_df['error_type'].value_counts().to_dict()}")
        
        return misclassified_df
    
    def generate_evaluation_summary(self) -> Dict[str, Any]:
        """Generate comprehensive evaluation summary"""
        if not self.results:
            return {}
        
        summary = {
            'overall_performance': {
                'accuracy': self.results['accuracy'],
                'precision': self.results['precision'],
                'recall': self.results['recall'],
                'f1_score': self.results['f1_score']
            },
            'class_performance': {
                class_name: {
                    'precision': self.results['precision_per_class'][i],
                    'recall': self.results['recall_per_class'][i],
                    'f1_score': self.results['f1_per_class'][i]
                }
                for i, class_name in enumerate(self.class_names)
            },
            'confusion_matrix': self.results['confusion_matrix'].tolist(),
            'total_samples': len(self.results['y_true']),
            'misclassified_samples': np.sum(self.results['y_true'] != self.results['y_pred'])
        }
        
        return summary

def test_model_evaluator():
    """
    Test function for model evaluator
    """
    from model_trainer import ModelTrainer
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
    
    # Train model
    trainer = ModelTrainer()
    X, y = trainer.prepare_features(labeled_segments)
    X_balanced, y_balanced = trainer.handle_class_imbalance(X, y)
    training_results = trainer.train_model(X_balanced, y_balanced, use_hyperparameter_tuning=False)
    
    # Evaluate model
    evaluator = ModelEvaluator()
    evaluation_results = evaluator.evaluate_model(
        y_true=training_results['y_test'],
        y_pred=training_results['y_pred'],
        feature_importance=training_results['feature_importance']
    )
    
    # Print detailed report
    evaluator.print_detailed_report()
    
    # Analyze misclassifications
    misclassified_df = evaluator.analyze_misclassifications(training_results['X_test'])
    
    return evaluator, evaluation_results

if __name__ == "__main__":
    test_model_evaluator() 