#!/usr/bin/env python3
"""
Complete Model Pipeline Test for Toronto Road Segment Crash Risk Prediction

This script tests the complete machine learning pipeline from data processing
through model training and evaluation.
"""

import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_complete_pipeline():
    """Test the complete ML pipeline"""
    print("🚀 Testing Complete Machine Learning Pipeline")
    print("=" * 80)
    
    try:
        # Import all required modules
        from src.data_processing.data_loader import load_and_clean_data
        from src.data_processing.spatial_join_fast import perform_spatial_join_fast
        from src.feature_engineering.feature_creator import create_segment_features
        from src.feature_engineering.label_generator import generate_risk_labels
        from src.models.model_trainer import ModelTrainer
        from src.models.model_evaluator import ModelEvaluator
        
        print("\n" + "=" * 60)
        print("STEP 1: Data Processing Pipeline")
        print("=" * 60)
        
        # Load and process data
        data_dir = Path("data")
        collision_data, ksi_data, road_network = load_and_clean_data(data_dir)
        segment_crashes = perform_spatial_join_fast(collision_data, ksi_data, road_network)
        segment_features = create_segment_features(segment_crashes, road_network)
        labeled_segments = generate_risk_labels(segment_features)
        
        print(f"✅ Data processing completed!")
        print(f"   - Total segments: {len(labeled_segments)}")
        print(f"   - Features created: {len(labeled_segments.columns)}")
        print(f"   - Risk distribution: {labeled_segments['risk_label'].value_counts().to_dict()}")
        
        print("\n" + "=" * 60)
        print("STEP 2: Model Training")
        print("=" * 60)
        
        # Initialize and train model
        trainer = ModelTrainer()
        X, y = trainer.prepare_features(labeled_segments)
        X_balanced, y_balanced = trainer.handle_class_imbalance(X, y)
        
        print(f"✅ Feature preparation completed!")
        print(f"   - Features: {len(X.columns)}")
        print(f"   - Original samples: {len(X)}")
        print(f"   - Balanced samples: {len(X_balanced)}")
        
        # Train model (without hyperparameter tuning for speed)
        training_results = trainer.train_model(X_balanced, y_balanced, use_hyperparameter_tuning=False)
        
        print(f"✅ Model training completed!")
        print(f"   - Test accuracy: {training_results['accuracy']:.4f}")
        print(f"   - CV F1-score: {training_results['cv_mean']:.4f} (+/- {training_results['cv_std'] * 2:.4f})")
        
        print("\n" + "=" * 60)
        print("STEP 3: Model Evaluation")
        print("=" * 60)
        
        # Evaluate model
        evaluator = ModelEvaluator()
        evaluation_results = evaluator.evaluate_model(
            y_true=training_results['y_test'],
            y_pred=training_results['y_pred'],
            feature_importance=training_results['feature_importance']
        )
        
        # Print detailed evaluation report
        evaluator.print_detailed_report()
        
        print("\n" + "=" * 60)
        print("STEP 4: Feature Importance Analysis")
        print("=" * 60)
        
        # Show top feature importance
        feature_importance = training_results['feature_importance']
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
        
        print(f"🎯 TOP 10 FEATURE IMPORTANCE:")
        for i, (feature, importance) in enumerate(top_features, 1):
            print(f"   {i:2d}. {feature}: {importance:.4f}")
        
        print("\n" + "=" * 60)
        print("STEP 5: Misclassification Analysis")
        print("=" * 60)
        
        # Analyze misclassifications
        misclassified_df = evaluator.analyze_misclassifications(training_results['X_test'])
        
        if not misclassified_df.empty:
            print(f"🔍 MISCLASSIFICATION ANALYSIS:")
            print(f"   - Total misclassified: {len(misclassified_df)}")
            print(f"   - Error types: {misclassified_df['error_type'].value_counts().to_dict()}")
        else:
            print(f"🎉 No misclassifications found!")
        
        print("\n" + "=" * 80)
        print("🎉 COMPLETE PIPELINE SUCCESS!")
        print("=" * 80)
        print("The machine learning pipeline is working correctly!")
        print("\nKey Results:")
        print(f"  📊 Model Accuracy: {training_results['accuracy']:.4f}")
        print(f"  🎯 CV F1-Score: {training_results['cv_mean']:.4f}")
        print(f"  🔍 Top Feature: {top_features[0][0]} ({top_features[0][1]:.4f})")
        print(f"  📈 Total Segments: {len(labeled_segments)}")
        print(f"  🚨 High Risk Segments: {len(labeled_segments[labeled_segments['risk_label'] == 'high'])}")
        
        print("\nNext Steps:")
        print("1. Implement visualization modules")
        print("2. Create interactive risk maps")
        print("3. Generate analysis reports")
        print("4. Deploy production model")
        
        return {
            'trainer': trainer,
            'evaluator': evaluator,
            'training_results': training_results,
            'evaluation_results': evaluation_results,
            'labeled_segments': labeled_segments
        }
        
    except Exception as e:
        print(f"❌ Pipeline failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main function to run the complete pipeline test"""
    results = test_complete_pipeline()
    
    if results:
        print(f"\n✅ Pipeline completed successfully!")
        print(f"Ready for visualization and deployment!")
    else:
        print(f"\n❌ Pipeline failed!")
        print(f"Please check the error messages above.")

if __name__ == "__main__":
    main() 