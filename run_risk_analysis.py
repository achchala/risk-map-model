#!/usr/bin/env python3
"""
Optimized visualization script that handles all road segments efficiently

This script creates:
1. Interactive HTML risk map (all segments, optimized)
2. Risk analysis dashboard
3. Data exports for QGIS
4. Summary report
"""

import sys
from pathlib import Path
import logging
import time
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from src.data_processing.data_loader import load_and_clean_data
from src.data_processing.spatial_join_fast import perform_spatial_join_fast
from src.feature_engineering.feature_creator import create_segment_features
from src.feature_engineering.label_generator import generate_risk_labels
from src.models.model_trainer import ModelTrainer
from src.visualization.risk_mapper import RiskMapper

def setup_logging():
    """Setup logging configuration"""
    # Only log to file, not console
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('visualization_pipeline.log')
        ]
    )

def main():
    """Main function to create all visualizations with optimized performance"""
    print("Toronto Road Risk Prediction - Optimized Visualization Pipeline")
    print("=" * 65)
    
    start_time = time.time()
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Step 1: Load and process data
        print("\nSTEP 1: Loading and Processing Data")
        print("-" * 40)
        
        data_dir = Path("data")
        logger.info("Loading collision data, KSI data, and road network...")
        
        collision_data, ksi_data, road_network = load_and_clean_data(data_dir)
        
        print(f"  Loaded {len(collision_data):,} collision records")
        print(f"  Loaded {len(ksi_data):,} KSI records")
        print(f"  Loaded {len(road_network):,} road segments")
        
        # Step 2: Spatial join
        print("\nSTEP 2: Performing Spatial Join")
        print("-" * 40)
        
        logger.info("Performing spatial join between crashes and road segments...")
        segment_crashes = perform_spatial_join_fast(collision_data, ksi_data, road_network)
        
        segments_with_crashes = len(segment_crashes[segment_crashes['num_total_crashes'] > 0])
        print(f"  Found {segments_with_crashes:,} segments with crashes")
        print(f"  Total segments to process: {len(segment_crashes):,}")
        
        # Step 3: Feature engineering
        print("\nSTEP 3: Feature Engineering")
        print("-" * 40)
        
        logger.info("Creating segment-level features...")
        features = create_segment_features(segment_crashes, road_network)
        
        print(f"  Created {len(features.columns)} features")
        
        # Step 4: Risk labeling
        print("\nSTEP 4: Risk Labeling")
        print("-" * 40)
        
        logger.info("Generating risk labels for road segments...")
        labeled_data = generate_risk_labels(features)
        
        risk_counts = labeled_data['risk_label'].value_counts()
        print(f"  Risk distribution:")
        print(f"    - Low: {risk_counts.get('low', 0):,} segments")
        print(f"    - Medium: {risk_counts.get('medium', 0):,} segments")
        print(f"    - High: {risk_counts.get('high', 0):,} segments")
        
        # Step 5: Model training (for predictions)
        print("\nSTEP 5: Model Training")
        print("-" * 40)
        
        logger.info("Training machine learning model...")
        trainer = ModelTrainer()
        X, y = trainer.prepare_features(labeled_data)
        X_balanced, y_balanced = trainer.handle_class_imbalance(X, y)
        results = trainer.train_model(X_balanced, y_balanced, use_hyperparameter_tuning=False)
        
        print(f"  Model accuracy: {results['accuracy']:.1%}")
        print(f"  Cross-validation F1: {results['cv_mean']:.1%}")
        
        # Step 6: Create visualizations
        print("\nSTEP 6: Creating Visualizations")
        print("-" * 40)
        
        mapper = RiskMapper()
        
        # Create interactive map (optimized for all segments)
        print("  Creating interactive risk map (all segments)...")
        logger.info("Creating interactive risk map...")
        map_file = mapper.create_interactive_risk_map(labeled_data, model_trainer=trainer)
        print(f"    Map saved to: {map_file}")
        
        # Create dashboard
        print("  Creating analysis dashboard...")
        logger.info("Creating risk analysis dashboard...")
        dashboard_file = mapper.create_risk_analysis_dashboard(labeled_data, results)
        print(f"    Dashboard saved to: {dashboard_file}")
        
        # Export data for QGIS
        print("  Exporting data for QGIS...")
        logger.info("Exporting data for QGIS...")
        exported_files = mapper.export_data_for_qgis(labeled_data)
        print(f"    GeoJSON files saved to: {exported_files['geojson']}")
        print(f"    CSV file saved to: {exported_files['csv']}")
        
        # Save trained model
        print("  Saving trained model...")
        logger.info("Saving trained model...")
        models_dir = Path("outputs/models")
        models_dir.mkdir(exist_ok=True)
        model_file = models_dir / "toronto_risk_model.joblib"
        trainer.save_model(str(model_file))
        print(f"    Model saved to: {model_file}")
        
        # Create summary report
        print("  Creating summary report...")
        logger.info("Creating summary report...")
        summary_report = create_summary_report(labeled_data, results, exported_files)
        print(f"    Report saved to: {summary_report}")
        
        # Final summary
        elapsed_time = time.time() - start_time
        print(f"\n" + "=" * 65)
        print(f"PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"Total time: {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
        print(f"=" * 65)
        
        print(f"\nOUTPUT FILES:")
        print(f"  Interactive Map: {map_file}")
        print(f"  Analysis Dashboard: {dashboard_file}")
        print(f"  Summary Report: {summary_report}")
        print(f"  Full Dataset (GeoJSON): {exported_files['geojson']}")
        print(f"  High-Risk Roads (GeoJSON): {exported_files['high_risk_geojson']}")
        print(f"  Road List (CSV): {exported_files['csv']}")
        print(f"  Trained Model: {model_file}")
        
        print(f"\nNEXT STEPS:")
        print(f"  1. Open the HTML files in your web browser")
        print(f"  2. Import GeoJSON files into QGIS for advanced analysis")
        print(f"  3. Use CSV files for statistical analysis in Excel")
        print(f"  4. Review the summary report for key findings")
        
        print(f"\nMVP STATUS: COMPLETE!")
        print(f"All visualizations created successfully with {len(labeled_data):,} road segments.")
        
        return True
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        print(f"\nERROR: {str(e)}")
        print("Check the log file for detailed error information.")
        return False

def create_summary_report(labeled_data, model_results, exported_files):
    """Create a comprehensive summary report"""
    from datetime import datetime
    
    output_dir = Path("outputs")
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(exist_ok=True)
    report_file = reports_dir / "summary_report.html"
    
    # Calculate key statistics
    total_segments = len(labeled_data)
    risk_counts = labeled_data['risk_label'].value_counts()
    high_risk = labeled_data[labeled_data['risk_label'] == 'high']
    segments_with_crashes = labeled_data[labeled_data['num_total_crashes'] > 0]
    
    # Calculate model performance metrics
    model_performance = ""
    if 'confusion_matrix' in model_results:
        cm = model_results['confusion_matrix']
        precision = model_results.get('precision', [0, 0, 0])
        recall = model_results.get('recall', [0, 0, 0])
        f1 = model_results.get('f1', [0, 0, 0])
        confidence_analysis = model_results.get('confidence_analysis', {})
        
        # Calculate practical metrics
        if len(cm) == 3:
            true_positives = cm[2, 2]  # High risk correctly identified
            false_positives = cm[0, 2] + cm[1, 2]  # Low/Medium predicted as High
            false_negatives = cm[2, 0] + cm[2, 1]  # High predicted as Low/Medium
            
            high_risk_accuracy = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            false_alarm_rate = false_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        else:
            high_risk_accuracy = 0
            false_alarm_rate = 0
        
        model_performance = f"""
        <div class="section">
            <h2>Model Performance & Reliability</h2>
            <div class="highlight">
                <p><strong>Overall Accuracy:</strong> {model_results['accuracy']:.1%}</p>
                <p><strong>High-Risk Detection Rate:</strong> {high_risk_accuracy:.1%}</p>
                <p><strong>False Alarm Rate:</strong> {false_alarm_rate:.1%}</p>
                <p><strong>Cross-Validation F1-Score:</strong> {model_results['cv_mean']:.1%}</p>
            </div>
            
            <h3>Per-Class Performance</h3>
            <table>
                <tr>
                    <th>Risk Level</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>F1-Score</th>
                </tr>
                <tr>
                    <td>Low Risk</td>
                    <td>{precision[0]:.3f}</td>
                    <td>{recall[0]:.3f}</td>
                    <td>{f1[0]:.3f}</td>
                </tr>
                <tr>
                    <td>Medium Risk</td>
                    <td>{precision[1]:.3f}</td>
                    <td>{recall[1]:.3f}</td>
                    <td>{f1[1]:.3f}</td>
                </tr>
                <tr>
                    <td>High Risk</td>
                    <td>{precision[2]:.3f}</td>
                    <td>{recall[2]:.3f}</td>
                    <td>{f1[2]:.3f}</td>
                </tr>
            </table>
            
            <h3>Model Reliability Assessment</h3>
            <ul>
                <li><strong>High-Risk Detection:</strong> The model correctly identifies {high_risk_accuracy:.1%} of actual high-risk segments</li>
                <li><strong>False Alarms:</strong> {false_alarm_rate:.1%} of segments predicted as high-risk are actually low or medium risk</li>
                <li><strong>Confidence Analysis:</strong> When the model is wrong, its average confidence is {confidence_analysis.get('confidence_when_wrong', 0):.1%}</li>
                <li><strong>High-Confidence Errors:</strong> {confidence_analysis.get('high_confidence_errors', 0)} predictions with >80% confidence were incorrect</li>
            </ul>
        </div>
        """
    
    report_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Toronto Road Risk Prediction - Summary Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
            .header {{ text-align: center; margin-bottom: 40px; }}
            .section {{ margin: 30px 0; padding: 20px; background-color: #f9f9f9; border-radius: 8px; }}
            .highlight {{ background-color: #e3f2fd; padding: 15px; border-radius: 5px; margin: 10px 0; }}
            .file-link {{ color: #1976d2; text-decoration: none; }}
            .file-link:hover {{ text-decoration: underline; }}
            table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
            th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #f5f5f5; font-weight: bold; }}
            .success {{ color: #2e7d32; font-weight: bold; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Toronto Road Risk Prediction - Summary Report</h1>
            <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p class="success">MVP Status: COMPLETE</p>
        </div>
        
        <div class="section">
            <h2>Executive Summary</h2>
            <div class="highlight">
                <p><strong>Total Road Segments Analyzed:</strong> {total_segments:,}</p>
                <p><strong>Segments with Historical Crashes:</strong> {len(segments_with_crashes):,} ({len(segments_with_crashes)/total_segments*100:.1f}%)</p>
                <p><strong>High-Risk Segments Identified:</strong> {len(high_risk):,} ({len(high_risk)/total_segments*100:.1f}%)</p>
                <p><strong>Model Accuracy:</strong> {model_results['accuracy']:.1%}</p>
                <p><strong>Cross-Validation F1-Score:</strong> {model_results['cv_mean']:.1%}</p>
            </div>
        </div>
        
        <div class="section">
            <h2>Risk Distribution</h2>
            <table>
                <tr>
                    <th>Risk Level</th>
                    <th>Count</th>
                    <th>Percentage</th>
                    <th>Description</th>
                </tr>
                <tr>
                    <td>Low Risk</td>
                    <td>{risk_counts.get('low', 0):,}</td>
                    <td>{risk_counts.get('low', 0)/total_segments*100:.1f}%</td>
                    <td>Segments with minimal crash history</td>
                </tr>
                <tr>
                    <td>Medium Risk</td>
                    <td>{risk_counts.get('medium', 0):,}</td>
                    <td>{risk_counts.get('medium', 0)/total_segments*100:.1f}%</td>
                    <td>Segments with moderate crash patterns</td>
                </tr>
                <tr>
                    <td>High Risk</td>
                    <td>{risk_counts.get('high', 0):,}</td>
                    <td>{risk_counts.get('high', 0)/total_segments*100:.1f}%</td>
                    <td>Segments requiring immediate attention</td>
                </tr>
            </table>
        </div>
        
        <div class="section">
            <h2>High-Risk Segment Analysis</h2>
            <p><strong>Average Length:</strong> {high_risk['segment_length'].mean():.1f}m</p>
            <p><strong>Average Total Crashes:</strong> {high_risk['num_total_crashes'].mean():.1f}</p>
            <p><strong>Average KSI Crashes:</strong> {high_risk['num_ksi_crashes'].mean():.1f}</p>
            <p><strong>Most Common Road Classes:</strong></p>
            <ul>
                {''.join([f'<li>{road_class}: {count} segments</li>' for road_class, count in high_risk['ROAD_CLASS'].value_counts().head(5).items()])}
            </ul>
        </div>
        
        {model_performance}
        
        <div class="section">
            <h2>Generated Files</h2>
            <table>
                <tr>
                    <th>File Type</th>
                    <th>Description</th>
                    <th>File Path</th>
                </tr>
                <tr>
                    <td>Interactive Map</td>
                    <td>HTML file with all road segments and risk levels</td>
                    <td><a href="toronto_risk_map.html" class="file-link">toronto_risk_map.html</a></td>
                </tr>
                <tr>
                    <td>Analysis Dashboard</td>
                    <td>HTML file with charts and detailed statistics</td>
                    <td><a href="risk_analysis_dashboard.html" class="file-link">risk_analysis_dashboard.html</a></td>
                </tr>
                <tr>
                    <td>Full Dataset (GeoJSON)</td>
                    <td>Complete dataset for QGIS import</td>
                    <td><a href="toronto_road_risk.geojson" class="file-link">toronto_road_risk.geojson</a></td>
                </tr>
                <tr>
                    <td>High-Risk Roads (GeoJSON)</td>
                    <td>High-risk segments only for QGIS</td>
                    <td><a href="toronto_high_risk_roads.geojson" class="file-link">toronto_high_risk_roads.geojson</a></td>
                </tr>
                <tr>
                    <td>Road List (CSV)</td>
                    <td>Tabular data for Excel analysis</td>
                    <td><a href="road_risk_list.csv" class="file-link">road_risk_list.csv</a></td>
                </tr>
            </table>
        </div>
        
        <div class="section">
            <h2>Key Findings</h2>
            <ul>
                <li><strong>{len(high_risk):,} high-risk segments</strong> identified across Toronto</li>
                <li><strong>{len(segments_with_crashes):,} segments</strong> have historical crash data</li>
                <li>Model achieves <strong>{model_results['accuracy']:.1%} accuracy</strong> in risk prediction</li>
                <li>Cross-validation F1-score of <strong>{model_results['cv_mean']:.1%}</strong> indicates good generalization</li>
            </ul>
        </div>
        
        <div class="section">
            <h2>Next Steps</h2>
            <ul>
                <li>Open the interactive map in your web browser to explore risk patterns</li>
                <li>Review the analysis dashboard for detailed statistics and trends</li>
                <li>Import GeoJSON files into QGIS for advanced spatial analysis</li>
                <li>Use the CSV files for further statistical analysis in Excel</li>
                <li>Focus on the {len(high_risk):,} high-risk segments for immediate intervention</li>
                <li>Consider model optimization for improved high-risk recall</li>
            </ul>
        </div>
        
        <div class="section">
            <h2>Technical Notes</h2>
            <ul>
                <li>All {total_segments:,} road segments included in the analysis</li>
                <li>ML predictions shown for high-risk segments and segments with crashes</li>
                <li>Interactive map optimized for performance with batch processing</li>
                <li>Data exported in multiple formats for different analysis tools</li>
            </ul>
        </div>
    </body>
    </html>
    """
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_html)
    
    return str(report_file)

if __name__ == "__main__":
    success = main()
    if not success:
        print("Visualization pipeline failed. Check the log file for details.") 