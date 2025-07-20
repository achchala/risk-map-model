"""
Toronto Road Segment Crash Risk Prediction - Main Pipeline

This script orchestrates the complete pipeline from raw data to risk predictions:
1. Data processing and cleaning
2. Spatial join of crashes to road segments
3. Feature engineering
4. Model training and evaluation
5. Risk prediction and output generation
"""

import os
import sys
import logging
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

from data_processing.data_loader import load_and_clean_data
from data_processing.spatial_join import perform_spatial_join
from feature_engineering.feature_creator import create_segment_features
from feature_engineering.label_generator import generate_risk_labels
from modeling.model_trainer import train_risk_model
from modeling.model_evaluator import evaluate_model
from visualization.map_generator import create_risk_map
from visualization.report_generator import generate_analysis_report

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Main pipeline execution"""
    logger.info("Starting Toronto Road Segment Crash Risk Prediction Pipeline")
    
    # Define paths
    data_dir = Path("data")
    outputs_dir = Path("outputs")
    models_dir = Path("models")
    
    # Ensure output directories exist
    outputs_dir.mkdir(exist_ok=True)
    models_dir.mkdir(exist_ok=True)
    
    try:
        # Step 1: Load and clean data
        logger.info("Step 1: Loading and cleaning data...")
        collision_data, ksi_data, road_network = load_and_clean_data(data_dir)
        
        # Step 2: Perform spatial join
        logger.info("Step 2: Performing spatial join...")
        segment_crashes = perform_spatial_join(collision_data, ksi_data, road_network)
        
        # Step 3: Create features
        logger.info("Step 3: Creating segment features...")
        segment_features = create_segment_features(segment_crashes, road_network)
        
        # Step 4: Generate risk labels
        logger.info("Step 4: Generating risk labels...")
        labeled_segments = generate_risk_labels(segment_features)
        
        # Step 5: Train model
        logger.info("Step 5: Training risk prediction model...")
        model, feature_names = train_risk_model(labeled_segments)
        
        # Step 6: Evaluate model
        logger.info("Step 6: Evaluating model performance...")
        evaluation_results = evaluate_model(model, labeled_segments, feature_names)
        
        # Step 7: Generate predictions and outputs
        logger.info("Step 7: Generating risk predictions and outputs...")
        risk_predictions = model.predict_proba(labeled_segments[feature_names])
        labeled_segments['risk_score'] = risk_predictions[:, 2]  # Probability of high risk
        
        # Step 8: Create visualizations
        logger.info("Step 8: Creating visualizations...")
        create_risk_map(labeled_segments, outputs_dir / "maps")
        
        # Step 9: Generate reports
        logger.info("Step 9: Generating analysis report...")
        generate_analysis_report(labeled_segments, evaluation_results, outputs_dir / "reports")
        
        # Step 10: Save model and results
        logger.info("Step 10: Saving model and results...")
        import joblib
        joblib.dump(model, models_dir / "risk_model.joblib")
        labeled_segments.to_file(outputs_dir / "maps" / "risk_segments.geojson", driver="GeoJSON")
        
        logger.info("Pipeline completed successfully!")
        logger.info(f"Results saved to: {outputs_dir}")
        logger.info(f"Model saved to: {models_dir}")
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {str(e)}")
        raise

if __name__ == "__main__":
    main() 