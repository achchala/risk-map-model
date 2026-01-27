"""
Pytest fixtures for backend API tests
"""

import pytest
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import LineString, Point
from unittest.mock import Mock, MagicMock
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def app_client():
    """Create Flask test client"""
    # Import app here to avoid loading model/data at module level
    import sys
    sys.path.insert(0, str(PROJECT_ROOT / "backend-api"))
    from app import app as flask_app

    flask_app.config['TESTING'] = True
    with flask_app.test_client() as client:
        yield client


@pytest.fixture
def mock_model_trainer():
    """Mock ModelTrainer for testing predictions"""
    mock_trainer = Mock()
    mock_trainer.model = Mock()
    mock_trainer.scaler = Mock()

    # Mock predict_proba to return realistic probabilities
    mock_trainer.model.predict_proba.return_value = np.array([[0.1, 0.3, 0.6]])  # low, medium, high

    # Mock scaler transform
    mock_trainer.scaler.transform.return_value = np.array([[1.0, 2.0, 3.0]])

    return mock_trainer


@pytest.fixture
def mock_road_segment():
    """Create a mock road segment as a GeoDataFrame row"""
    data = {
        'LINEAR_NAME': 'Test Street',
        'geometry': LineString([(43.65, -79.38), (43.66, -79.37)]),
        'total_crashes': 10,
        'injury_collisions': 5,
        'fatalities': 2,
        'ksi_count': 3,
        'risk_label': 'high',
        'confidence': 0.85,
        'crash_density': 0.5
    }
    return pd.Series(data)


@pytest.fixture
def mock_gdf_data():
    """Create mock GeoDataFrame with road segments"""
    data = {
        'LINEAR_NAME': ['King Street', 'Queen Street', 'Dundas Street'],
        'geometry': [
            LineString([(43.65, -79.38), (43.66, -79.37)]),
            LineString([(43.64, -79.39), (43.65, -79.38)]),
            LineString([(43.63, -79.40), (43.64, -79.39)])
        ],
        'total_crashes': [15, 8, 3],
        'injury_collisions': [10, 5, 2],
        'fatalities': [3, 1, 0],
        'ksi_count': [4, 2, 0],
        'risk_label': ['high', 'medium', 'low'],
        'confidence': [0.9, 0.75, 0.6],
        'crash_density': [0.8, 0.4, 0.1]
    }
    gdf = gpd.GeoDataFrame(data, crs="EPSG:4326")
    return gdf


@pytest.fixture
def valid_bbox_request():
    """Valid bounding box request data"""
    return {
        'north': 43.7,
        'south': 43.6,
        'east': -79.3,
        'west': -79.4
    }


@pytest.fixture
def valid_point_request():
    """Valid single point request data"""
    return {
        'latitude': 43.6532,
        'longitude': -79.3832
    }


@pytest.fixture
def mock_linestring():
    """Mock LineString geometry"""
    return LineString([(43.65, -79.38), (43.66, -79.37), (43.67, -79.36)])


@pytest.fixture
def mock_point():
    """Mock Point geometry"""
    return Point(43.6532, -79.3832)
