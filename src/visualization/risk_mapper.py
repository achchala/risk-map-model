"""
Risk mapping and visualization module for Toronto Road Segment Crash Risk Prediction

This module creates interactive and static risk maps, analysis reports, and data exports.
"""

import pandas as pd
import geopandas as gpd
import numpy as np
import folium
from folium import plugins
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class RiskMapper:
    """Class for creating risk maps and visualizations"""
    
    def __init__(self, output_dir: Path = None):
        """
        Initialize RiskMapper
        
        Args:
            output_dir: Directory to save outputs
        """
        self.output_dir = output_dir or Path("outputs")
        self.output_dir.mkdir(exist_ok=True)
        
        # Color scheme for risk levels
        self.risk_colors = {
            'low': '#2E8B57',      # Sea Green
            'medium': '#FFA500',   # Orange
            'high': '#DC143C'      # Crimson Red
        }
        
        # Toronto center coordinates
        self.toronto_center = [43.6532, -79.3832]
        
    def create_interactive_risk_map(self, risk_data: gpd.GeoDataFrame, 
                                   title: str = "Toronto Road Segment Risk Map",
                                   model_trainer=None) -> str:
        """
        Create an interactive HTML risk map
        
        Args:
            risk_data: GeoDataFrame with risk predictions and geometry
            title: Map title
            
        Returns:
            Path to the generated HTML file
        """
        logger.info("Creating interactive risk map...")
        
        # Create base map centered on Toronto
        m = folium.Map(
            location=self.toronto_center,
            zoom_start=11,
            tiles='OpenStreetMap'
        )
        
        # Add tile layers with proper attribution
        folium.TileLayer(
            tiles='cartodbpositron',
            name='Light Map',
            attr='CartoDB Positron'
        ).add_to(m)
        folium.TileLayer(
            tiles='cartodbdark_matter',
            name='Dark Map',
            attr='CartoDB Dark Matter'
        ).add_to(m)
        folium.TileLayer(
            tiles='Stamen Terrain',
            name='Terrain',
            attr='Stamen Design'
        ).add_to(m)
        
        # Create feature groups for each risk level
        risk_groups = {}
        for risk_level in ['low', 'medium', 'high']:
            risk_groups[risk_level] = folium.FeatureGroup(
                name=f'{risk_level.title()} Risk',
                overlay=True
            )
        
        # Add all road segments to map (optimized for performance)
        print(f"  Adding {len(risk_data)} segments to map...")
        
        # Process in batches for better performance
        batch_size = 1000
        for i in range(0, len(risk_data), batch_size):
            batch = risk_data.iloc[i:i+batch_size]
            print(f"    Processing batch {i//batch_size + 1}/{(len(risk_data)-1)//batch_size + 1}...")
            
            for idx, row in batch.iterrows():
                if pd.isna(row.geometry) or row.geometry.is_empty:
                    continue
                    
                # Get risk level and color
                risk_level = row.get('risk_label', 'low')
                color = self.risk_colors.get(risk_level, '#808080')
                
                # Create popup content with predicted risk if model is available
                popup_content = self._create_popup_content(row, model_trainer)
                
                # Add to map based on geometry type
                if row.geometry.geom_type == 'LineString':
                    folium.PolyLine(
                        locations=[[coord[1], coord[0]] for coord in row.geometry.coords],
                        color=color,
                        weight=3,
                        opacity=0.8,
                        popup=folium.Popup(popup_content, max_width=300)
                    ).add_to(risk_groups[risk_level])
                elif row.geometry.geom_type == 'MultiLineString':
                    for line in row.geometry.geoms:
                        folium.PolyLine(
                            locations=[[coord[1], coord[0]] for coord in line.coords],
                            color=color,
                            weight=3,
                            opacity=0.8,
                            popup=folium.Popup(popup_content, max_width=300)
                        ).add_to(risk_groups[risk_level])
        
        # Add feature groups to map
        for group in risk_groups.values():
            group.add_to(m)
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        # Add legend
        legend_html = self._create_legend_html()
        m.get_root().html.add_child(folium.Element(legend_html))
        
        # Add title
        title_html = f'''
        <div style="position: fixed; 
                    top: 10px; left: 50px; width: 300px; height: 90px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; padding: 10px">
        <b>{title}</b><br>
        <i>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}</i><br>
        <i>Total Segments: {len(risk_data)}</i>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(title_html))
        
        # Save map
        maps_dir = self.output_dir / "maps"
        maps_dir.mkdir(exist_ok=True)
        output_file = maps_dir / "toronto_risk_map.html"
        m.save(str(output_file))
        
        logger.info(f"Interactive risk map saved to {output_file}")
        return str(output_file)
    
    def _create_popup_content(self, row: pd.Series, model_trainer=None) -> str:
        """Create popup content for a road segment"""
        road_name = row.get('LINEAR_NAME', 'Unknown')
        risk_level = row.get('risk_label', 'Unknown')
        road_class = row.get('ROAD_CLASS', 'Unknown')
        length = row.get('segment_length', 0)
        crashes = row.get('num_total_crashes', 0)
        ksi = row.get('num_ksi_crashes', 0)
        
        # Get predicted risk if model is available (only for high-risk or crash segments to save time)
        predicted_info = ""
        if model_trainer is not None and (risk_level == 'high' or crashes > 0):
            try:
                # Prepare features for this segment
                segment_df = pd.DataFrame([row])
                X, _ = model_trainer.prepare_features(segment_df)
                
                # Get prediction and probabilities
                prediction = model_trainer.predict(X)
                probabilities = model_trainer.model.predict_proba(X)[0]
                
                # Map prediction to label
                label_map = {0: 'low', 1: 'medium', 2: 'high'}
                predicted_label = label_map.get(int(prediction[0]), 'Unknown')
                
                # Get confidence (max probability)
                confidence = max(probabilities) * 100
                
                predicted_info = f"""
                <b>Predicted Risk:</b> {predicted_label.upper()}<br>
                <b>Confidence:</b> {confidence:.1f}%<br>
                <b>Probabilities:</b> Low: {probabilities[0]*100:.1f}%, Medium: {probabilities[1]*100:.1f}%, High: {probabilities[2]*100:.1f}%<br>
                """
            except Exception as e:
                predicted_info = f"<b>Prediction Error:</b> {str(e)}<br>"
        
        popup_html = f"""
        <div style="width: 280px;">
            <h4>{road_name}</h4>
            <b>Actual Risk Level:</b> {risk_level.upper()}<br>
            {predicted_info}
            <b>Road Class:</b> {road_class}<br>
            <b>Length:</b> {length:.1f}m<br>
            <b>Historical Crashes:</b> {crashes} total, {ksi} KSI<br>
        </div>
        """
        return popup_html
    
    def _create_legend_html(self) -> str:
        """Create legend HTML"""
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 150px; height: 120px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; padding: 10px">
        <p><b>Risk Levels</b></p>
        <p><i class="fa fa-circle" style="color:#2E8B57"></i> Low Risk</p>
        <p><i class="fa fa-circle" style="color:#FFA500"></i> Medium Risk</p>
        <p><i class="fa fa-circle" style="color:#DC143C"></i> High Risk</p>
        </div>
        '''
        return legend_html
    
    def create_risk_analysis_dashboard(self, risk_data: gpd.GeoDataFrame, model_results: dict = None) -> str:
        """
        Create a comprehensive risk analysis dashboard
        
        Args:
            risk_data: GeoDataFrame with risk predictions
            model_results: Dictionary containing model performance metrics
            
        Returns:
            Path to the generated HTML dashboard
        """
        logger.info("Creating risk analysis dashboard...")
        
        # Calculate statistics
        stats = self._calculate_risk_statistics(risk_data)
        
        # Create HTML dashboard
        dashboard_html = self._generate_dashboard_html(risk_data, stats, model_results)
        
        # Save dashboard
        reports_dir = self.output_dir / "reports"
        reports_dir.mkdir(exist_ok=True)
        output_file = reports_dir / "risk_analysis_dashboard.html"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(dashboard_html)
        
        logger.info(f"Risk analysis dashboard saved to {output_file}")
        return str(output_file)
    
    def _calculate_risk_statistics(self, risk_data: gpd.GeoDataFrame) -> dict:
        """Calculate comprehensive risk statistics"""
        stats = {}
        
        # Risk distribution
        risk_counts = risk_data['risk_label'].value_counts()
        stats['risk_distribution'] = {
            'low': int(risk_counts.get('low', 0)),
            'medium': int(risk_counts.get('medium', 0)),
            'high': int(risk_counts.get('high', 0))
        }
        
        # Risk percentages
        total_segments = len(risk_data)
        stats['risk_percentages'] = {
            'low': (stats['risk_distribution']['low'] / total_segments) * 100,
            'medium': (stats['risk_distribution']['medium'] / total_segments) * 100,
            'high': (stats['risk_distribution']['high'] / total_segments) * 100
        }
        
        # Road class analysis
        road_class_risk = risk_data.groupby(['ROAD_CLASS', 'risk_label']).size().unstack(fill_value=0)
        stats['road_class_risk'] = road_class_risk.to_dict()
        
        # High-risk segments analysis
        high_risk = risk_data[risk_data['risk_label'] == 'high']
        stats['high_risk_analysis'] = {
            'count': len(high_risk),
            'avg_length': high_risk['segment_length'].mean(),
            'avg_crashes': high_risk['num_total_crashes'].mean(),
            'avg_ksi': high_risk['num_ksi_crashes'].mean()
        }
        
        # Top high-risk roads
        top_high_risk = high_risk.nlargest(10, 'num_total_crashes')[['LINEAR_NAME', 'num_total_crashes', 'num_ksi_crashes']]
        stats['top_high_risk_roads'] = top_high_risk.to_dict('records')
        
        return stats
    
    def _generate_dashboard_html(self, risk_data: gpd.GeoDataFrame, stats: dict, model_results: dict = None) -> str:
        """Generate HTML dashboard content"""
        
        # Create charts using Plotly (embedded)
        charts_html = self._create_charts_html(stats, model_results)
        
        dashboard_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Toronto Road Risk Analysis Dashboard</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
                .container {{ max-width: 1400px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 10px; }}
                .header {{ text-align: center; margin-bottom: 30px; }}
                .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 30px; }}
                .stat-card {{ background-color: #f8f9fa; padding: 20px; border-radius: 8px; text-align: center; }}
                .stat-number {{ font-size: 2em; font-weight: bold; color: #007bff; }}
                .chart-container {{ margin: 30px 0; text-align: center; }}
                .chart-container h2 {{ margin-bottom: 20px; color: #333; }}
                .risk-high {{ color: #DC143C; font-weight: bold; }}
                .risk-medium {{ color: #FFA500; font-weight: bold; }}
                .risk-low {{ color: #2E8B57; font-weight: bold; }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f8f9fa; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1> Toronto Road Risk Analysis Dashboard</h1>
                    <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
                
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-number">{stats['risk_distribution']['low']:,}</div>
                        <div>Low Risk Segments</div>
                        <div class="risk-low">{stats['risk_percentages']['low']:.1f}%</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">{stats['risk_distribution']['medium']:,}</div>
                        <div>Medium Risk Segments</div>
                        <div class="risk-medium">{stats['risk_percentages']['medium']:.1f}%</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">{stats['risk_distribution']['high']:,}</div>
                        <div>High Risk Segments</div>
                        <div class="risk-high">{stats['risk_percentages']['high']:.1f}%</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">{len(risk_data):,}</div>
                        <div>Total Segments</div>
                    </div>
                </div>
                
                <div class="chart-container">
                    <h2> Risk Distribution</h2>
                    {charts_html}
                </div>
                
                <div class="chart-container">
                    <h2> Top High-Risk Roads</h2>
                    <table>
                        <tr>
                            <th>Road Name</th>
                            <th>Total Crashes</th>
                            <th>KSI Crashes</th>
                        </tr>
                        {''.join([f"<tr><td>{road['LINEAR_NAME']}</td><td>{road['num_total_crashes']}</td><td>{road['num_ksi_crashes']}</td></tr>" for road in stats['top_high_risk_roads']])}
                    </table>
                </div>
                
                <div class="chart-container">
                    <h2> High-Risk Segment Analysis</h2>
                    <p><strong>Average Length:</strong> {stats['high_risk_analysis']['avg_length']:.1f}m</p>
                    <p><strong>Average Total Crashes:</strong> {stats['high_risk_analysis']['avg_crashes']:.1f}</p>
                    <p><strong>Average KSI Crashes:</strong> {stats['high_risk_analysis']['avg_ksi']:.1f}</p>
                </div>
                
                {self._create_model_performance_section(model_results) if model_results else ''}
            </div>
        </body>
        </html>
        """
        
        return dashboard_html
    
    def _create_charts_html(self, stats: dict, model_results: dict = None) -> str:
        """Create embedded Plotly charts"""
        import json
        
        # Risk distribution pie chart
        pie_chart = {
            'data': [{
                'values': [stats['risk_distribution']['low'], 
                          stats['risk_distribution']['medium'], 
                          stats['risk_distribution']['high']],
                'labels': ['Low Risk', 'Medium Risk', 'High Risk'],
                'type': 'pie',
                'marker': {'colors': ['#2E8B57', '#FFA500', '#DC143C']}
            }],
            'layout': {
                'title': 'Risk Distribution',
                'height': 400
            }
        }
        
        # Risk percentages bar chart
        bar_chart = {
            'data': [{
                'x': ['Low Risk', 'Medium Risk', 'High Risk'],
                'y': [stats['risk_percentages']['low'], 
                     stats['risk_percentages']['medium'], 
                     stats['risk_percentages']['high']],
                'type': 'bar',
                'marker': {'color': ['#2E8B57', '#FFA500', '#DC143C']}
            }],
            'layout': {
                'title': 'Risk Percentages',
                'yaxis': {'title': 'Percentage (%)'},
                'height': 400
            }
        }
        
        charts_html = f"""
        <div id="pie-chart"></div>
        <div id="bar-chart"></div>
        """
        
        # Add model performance charts if available
        if model_results and 'confusion_matrix' in model_results:
            # Create confusion matrix heatmap
            cm = model_results['confusion_matrix']
            cm_data = [{
                'z': cm.tolist() if hasattr(cm, 'tolist') else cm,
                'x': ['Predicted Low', 'Predicted Medium', 'Predicted High'],
                'y': ['Actual Low', 'Actual Medium', 'Actual High'],
                'type': 'heatmap',
                'colorscale': 'Blues',
                'showscale': True
            }]
            
            cm_chart = {
                'data': cm_data,
                'layout': {
                    'title': 'Confusion Matrix',
                    'xaxis': {
                        'title': {
                            'text': 'Predicted',
                            'standoff': 20,
                            'font': {'size': 14}
                        },
                        'tickangle': 0,
                        'tickfont': {'size': 12},
                        'titlefont': {'size': 14}
                    },
                    'yaxis': {
                        'title': {
                            'text': 'Actual',
                            'standoff': 20,
                            'font': {'size': 14}
                        },
                        'tickangle': 0,
                        'tickfont': {'size': 12},
                        'titlefont': {'size': 14}
                    },
                    'height': 550,
                    'width': 650,
                    'margin': {
                        'l': 100,  # left margin - increased for y-axis label
                        'r': 80,   # right margin
                        't': 80,   # top margin
                        'b': 80    # bottom margin
                    }
                }
            }
            
            # Create per-class performance chart
            if 'precision' in model_results:
                precision = model_results['precision'].tolist() if hasattr(model_results['precision'], 'tolist') else model_results['precision']
                recall = model_results['recall'].tolist() if hasattr(model_results['recall'], 'tolist') else model_results['recall']
                f1 = model_results['f1'].tolist() if hasattr(model_results['f1'], 'tolist') else model_results['f1']
                
                perf_data = [{
                    'x': ['Low Risk', 'Medium Risk', 'High Risk'],
                    'y': precision,
                    'type': 'bar',
                    'name': 'Precision',
                    'marker': {'color': '#1f77b4'}
                }, {
                    'x': ['Low Risk', 'Medium Risk', 'High Risk'],
                    'y': recall,
                    'type': 'bar',
                    'name': 'Recall',
                    'marker': {'color': '#ff7f0e'}
                }, {
                    'x': ['Low Risk', 'Medium Risk', 'High Risk'],
                    'y': f1,
                    'type': 'bar',
                    'name': 'F1-Score',
                    'marker': {'color': '#2ca02c'}
                }]
                
                perf_chart = {
                    'data': perf_data,
                    'layout': {
                        'title': 'Per-Class Performance Metrics',
                        'yaxis': {'title': 'Score'},
                        'xaxis': {
                            'tickangle': 0,
                            'tickfont': {'size': 12}
                        },
                        'barmode': 'group',
                        'height': 500,
                        'width': 600,
                        'margin': {
                            'l': 80,
                            'r': 80,
                            't': 80,
                            'b': 80
                        }
                    }
                }
                
                charts_html += f"""
                <div style="display: flex; justify-content: space-around; flex-wrap: wrap; gap: 30px; margin: 30px 0;">
                    <div style="flex: 1; min-width: 600px;">
                        <h3 style="text-align: center; margin-bottom: 15px;">Confusion Matrix</h3>
                        <div id="confusion-matrix"></div>
                    </div>
                    <div style="flex: 1; min-width: 600px;">
                        <h3 style="text-align: center; margin-bottom: 15px;">Performance Metrics</h3>
                        <div id="performance-metrics"></div>
                    </div>
                </div>
                
                <script>
                    Plotly.newPlot('confusion-matrix', {json.dumps(cm_chart['data'])}, {json.dumps(cm_chart['layout'])});
                    Plotly.newPlot('performance-metrics', {json.dumps(perf_chart['data'])}, {json.dumps(perf_chart['layout'])});
                </script>
                """
        
        charts_html += f"""
        <script>
            Plotly.newPlot('pie-chart', {json.dumps(pie_chart['data'])}, {json.dumps(pie_chart['layout'])});
            Plotly.newPlot('bar-chart', {json.dumps(bar_chart['data'])}, {json.dumps(bar_chart['layout'])});
        </script>
        """
        
        return charts_html
    
    def _create_model_performance_section(self, model_results: dict) -> str:
        """Create HTML section for model performance metrics"""
        if not model_results or 'confusion_matrix' not in model_results:
            return ""
        
        # Extract metrics
        accuracy = model_results.get('accuracy', 0)
        cm = model_results.get('confusion_matrix', np.array([]))
        precision = model_results.get('precision', [])
        recall = model_results.get('recall', [])
        f1 = model_results.get('f1', [])
        confidence_analysis = model_results.get('confidence_analysis', {})
        
        # Convert numpy arrays to lists if needed
        if hasattr(precision, 'tolist'):
            precision = precision.tolist()
        if hasattr(recall, 'tolist'):
            recall = recall.tolist()
        if hasattr(f1, 'tolist'):
            f1 = f1.tolist()
        
        # Calculate practical metrics
        if len(cm) == 3:
            # Confusion matrix interpretation
            true_positives = cm[2, 2]  # High risk correctly identified
            false_positives = cm[0, 2] + cm[1, 2]  # Low/Medium predicted as High
            false_negatives = cm[2, 0] + cm[2, 1]  # High predicted as Low/Medium
            
            high_risk_accuracy = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            false_alarm_rate = false_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        else:
            high_risk_accuracy = 0
            false_alarm_rate = 0
        
        return f"""
        <div class="chart-container">
            <h2> Model Performance Analysis</h2>
            
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-number">{accuracy:.1%}</div>
                    <div>Overall Accuracy</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{high_risk_accuracy:.1%}</div>
                    <div>High-Risk Detection Rate</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{false_alarm_rate:.1%}</div>
                    <div>False Alarm Rate</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{confidence_analysis.get('mean_confidence', 0):.1%}</div>
                    <div>Average Confidence</div>
                </div>
            </div>
            
            <div style="margin: 20px 0; padding: 20px; background-color: #f8f9fa; border-radius: 8px;">
                <h3>Model Reliability Assessment</h3>
                <p><strong>High-Risk Detection:</strong> The model correctly identifies {high_risk_accuracy:.1%} of actual high-risk segments.</p>
                <p><strong>False Alarms:</strong> {false_alarm_rate:.1%} of segments predicted as high-risk are actually low or medium risk.</p>
                <p><strong>Confidence Analysis:</strong> When the model is wrong, its average confidence is {confidence_analysis.get('confidence_when_wrong', 0):.1%}.</p>
                <p><strong>High-Confidence Errors:</strong> {confidence_analysis.get('high_confidence_errors', 0)} predictions with >80% confidence were incorrect.</p>
            </div>
            
            <div style="margin: 20px 0;">
                <h3>Per-Class Performance</h3>
                <table>
                    <tr>
                        <th>Risk Level</th>
                        <th>Precision</th>
                        <th>Recall</th>
                        <th>F1-Score</th>
                        <th>Support</th>
                    </tr>
                    <tr>
                        <td>Low Risk</td>
                        <td>{precision[0]:.3f}</td>
                        <td>{recall[0]:.3f}</td>
                        <td>{f1[0]:.3f}</td>
                        <td>{model_results.get('support', [0,0,0])[0]:,}</td>
                    </tr>
                    <tr>
                        <td>Medium Risk</td>
                        <td>{precision[1]:.3f}</td>
                        <td>{recall[1]:.3f}</td>
                        <td>{f1[1]:.3f}</td>
                        <td>{model_results.get('support', [0,0,0])[1]:,}</td>
                    </tr>
                    <tr>
                        <td>High Risk</td>
                        <td>{precision[2]:.3f}</td>
                        <td>{recall[2]:.3f}</td>
                        <td>{f1[2]:.3f}</td>
                        <td>{model_results.get('support', [0,0,0])[2]:,}</td>
                    </tr>
                </table>
            </div>
        </div>
        """
    
    def export_data_for_qgis(self, risk_data: gpd.GeoDataFrame) -> dict:
        """
        Export data in formats suitable for QGIS
        
        Args:
            risk_data: GeoDataFrame with risk predictions
            
        Returns:
            Dictionary with paths to exported files
        """
        logger.info("Exporting data for QGIS...")
        
        exported_files = {}
        
        # Export full dataset as GeoJSON
        maps_dir = self.output_dir / "maps"
        maps_dir.mkdir(exist_ok=True)
        geojson_file = maps_dir / "toronto_road_risk.geojson"
        risk_data.to_file(geojson_file, driver='GeoJSON')
        exported_files['geojson'] = str(geojson_file)
        
        # Export high-risk segments only
        high_risk = risk_data[risk_data['risk_label'] == 'high']
        high_risk_file = maps_dir / "toronto_high_risk_roads.geojson"
        high_risk.to_file(high_risk_file, driver='GeoJSON')
        exported_files['high_risk_geojson'] = str(high_risk_file)
        
        # Export summary statistics as CSV
        reports_dir = self.output_dir / "reports"
        reports_dir.mkdir(exist_ok=True)
        summary_stats = risk_data.groupby('risk_label').agg({
            'segment_length': ['count', 'mean', 'sum'],
            'num_total_crashes': ['sum', 'mean'],
            'num_ksi_crashes': ['sum', 'mean']
        }).round(2)
        
        csv_file = reports_dir / "risk_summary_statistics.csv"
        summary_stats.to_csv(csv_file)
        exported_files['summary_csv'] = str(csv_file)
        
        # Export road list as CSV
        road_list = risk_data[['LINEAR_NAME', 'ROAD_CLASS', 'risk_label', 
                              'segment_length', 'num_total_crashes', 'num_ksi_crashes']].copy()
        road_list_file = reports_dir / "road_risk_list.csv"
        road_list.to_csv(road_list_file, index=False)
        exported_files['csv'] = str(road_list_file)  # Use 'csv' key for compatibility
        
        logger.info(f"Data exported to {self.output_dir}")
        return exported_files

def test_risk_mapper():
    """Test function for risk mapping"""
    from data_processing.data_loader import load_and_clean_data
    from data_processing.spatial_join_fast import perform_spatial_join_fast
    from feature_engineering.feature_creator import create_segment_features
    from feature_engineering.label_generator import generate_risk_labels
    
    logging.basicConfig(level=logging.INFO)
    
    # Load and process data
    data_dir = Path("data")
    collision_data, ksi_data, road_network = load_and_clean_data(data_dir)
    segment_crashes = perform_spatial_join_fast(collision_data, ksi_data, road_network)
    features = create_segment_features(segment_crashes, road_network)
    labeled_data = generate_risk_labels(features)
    
    # Create visualizations
    mapper = RiskMapper()
    
    # Create interactive map
    map_file = mapper.create_interactive_risk_map(labeled_data)
    print(f"Interactive map: {map_file}")
    
    # Create dashboard
    dashboard_file = mapper.create_risk_analysis_dashboard(labeled_data)
    print(f"Dashboard: {dashboard_file}")
    
    # Export data
    exported_files = mapper.export_data_for_qgis(labeled_data)
    print(f"Exported files: {exported_files}")
    
    return mapper, labeled_data

if __name__ == "__main__":
    test_risk_mapper() 