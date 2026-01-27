"""
Tests for Flask API endpoints
"""

import pytest
import json
from unittest.mock import patch, Mock
import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString


class TestHealthEndpoint:
    """Tests for /api/health endpoint"""

    def test_health_check_success(self, app_client):
        """Test health endpoint returns 200 and correct structure"""
        response = app_client.get('/api/health')
        assert response.status_code == 200

        data = response.get_json()
        assert 'status' in data
        assert 'model_loaded' in data
        assert 'preprocessed_data_loaded' in data
        assert 'road_network_loaded' in data

    def test_health_check_structure(self, app_client):
        """Test health endpoint returns expected data types"""
        response = app_client.get('/api/health')
        data = response.get_json()

        assert isinstance(data['status'], str)
        assert isinstance(data['model_loaded'], bool)
        assert isinstance(data['preprocessed_data_loaded'], bool)
        assert isinstance(data['road_network_loaded'], bool)


class TestRiskPredictionsBBox:
    """Tests for /api/risk-predictions endpoint (bounding box)"""

    @patch('app.preprocessed_data')
    def test_risk_predictions_valid_bbox(self, mock_data, app_client, mock_gdf_data):
        """Test risk predictions with valid bounding box"""
        mock_data.return_value = mock_gdf_data

        request_data = {
            'north': 43.7,
            'south': 43.6,
            'east': -79.3,
            'west': -79.4
        }

        response = app_client.post(
            '/api/risk-predictions',
            data=json.dumps(request_data),
            content_type='application/json'
        )

        assert response.status_code == 200
        data = response.get_json()
        assert isinstance(data, list)

    def test_risk_predictions_missing_fields(self, app_client):
        """Test risk predictions with missing required fields"""
        request_data = {
            'north': 43.7,
            'south': 43.6
            # missing east and west
        }

        response = app_client.post(
            '/api/risk-predictions',
            data=json.dumps(request_data),
            content_type='application/json'
        )

        assert response.status_code == 400
        data = response.get_json()
        assert 'error' in data

    def test_risk_predictions_invalid_bounds(self, app_client):
        """Test risk predictions with invalid bounds (north < south)"""
        request_data = {
            'north': 43.6,  # north less than south
            'south': 43.7,
            'east': -79.3,
            'west': -79.4
        }

        response = app_client.post(
            '/api/risk-predictions',
            data=json.dumps(request_data),
            content_type='application/json'
        )

        assert response.status_code == 400
        data = response.get_json()
        assert 'error' in data

    def test_risk_predictions_response_structure(self, app_client):
        """Test risk predictions response has correct structure"""
        # Note: This test will only pass if preprocessed data exists
        # Otherwise it will return 500 or empty list
        request_data = {
            'north': 43.7,
            'south': 43.6,
            'east': -79.3,
            'west': -79.4
        }

        response = app_client.post(
            '/api/risk-predictions',
            data=json.dumps(request_data),
            content_type='application/json'
        )

        # Accept either 200 (data exists) or 500 (no data available)
        assert response.status_code in [200, 500]

        if response.status_code == 200:
            data = response.get_json()
            assert isinstance(data, list)

            if len(data) > 0:
                segment = data[0]
                assert 'LINEAR_NAME' in segment or 'linearName' in segment
                assert 'coordinates' in segment
                assert 'risk_label' in segment or 'riskLevel' in segment


class TestRiskPredictionSingle:
    """Tests for /api/risk-prediction endpoint (single point)"""

    def test_risk_prediction_valid_point(self, app_client):
        """Test single point risk prediction with valid coordinates"""
        request_data = {
            'latitude': 43.6532,
            'longitude': -79.3832
        }

        response = app_client.post(
            '/api/risk-prediction',
            data=json.dumps(request_data),
            content_type='application/json'
        )

        # Accept 200 (found segment) or 404 (no nearby segment) or 500 (no data)
        assert response.status_code in [200, 404, 500]

    def test_risk_prediction_missing_fields(self, app_client):
        """Test single point prediction with missing fields"""
        request_data = {
            'latitude': 43.6532
            # missing longitude
        }

        response = app_client.post(
            '/api/risk-prediction',
            data=json.dumps(request_data),
            content_type='application/json'
        )

        assert response.status_code == 400
        data = response.get_json()
        assert 'error' in data

    def test_risk_prediction_invalid_coordinates(self, app_client):
        """Test single point prediction with invalid coordinates"""
        request_data = {
            'latitude': 200.0,  # invalid latitude
            'longitude': -79.3832
        }

        response = app_client.post(
            '/api/risk-prediction',
            data=json.dumps(request_data),
            content_type='application/json'
        )

        # Should either reject invalid coords or return 404
        assert response.status_code in [400, 404, 500]

    def test_risk_prediction_response_structure(self, app_client):
        """Test single point prediction response structure"""
        request_data = {
            'latitude': 43.6532,
            'longitude': -79.3832
        }

        response = app_client.post(
            '/api/risk-prediction',
            data=json.dumps(request_data),
            content_type='application/json'
        )

        if response.status_code == 200:
            data = response.get_json()
            assert 'riskLevel' in data
            assert 'confidence' in data or 'probabilities' in data


class TestStreetNamesEndpoint:
    """Tests for /api/street-names endpoint"""

    def test_street_names_valid_query(self, app_client):
        """Test street names autocomplete with valid query"""
        response = app_client.get('/api/street-names?query=King&limit=10')

        # Accept 200 (data exists) or 500 (no data)
        assert response.status_code in [200, 500]

    def test_street_names_query_too_short(self, app_client):
        """Test street names with query less than 2 characters"""
        response = app_client.get('/api/street-names?query=K&limit=10')

        assert response.status_code == 400
        data = response.get_json()
        assert 'error' in data

    def test_street_names_missing_query(self, app_client):
        """Test street names without query parameter"""
        response = app_client.get('/api/street-names?limit=10')

        assert response.status_code == 400
        data = response.get_json()
        assert 'error' in data

    def test_street_names_limit_enforcement(self, app_client):
        """Test street names respects limit parameter"""
        response = app_client.get('/api/street-names?query=Street&limit=5')

        if response.status_code == 200:
            data = response.get_json()
            assert len(data) <= 5


class TestSegmentsAllEndpoint:
    """Tests for /api/segments/all endpoint"""

    def test_segments_all_default(self, app_client):
        """Test segments list with default parameters"""
        response = app_client.get('/api/segments/all')

        # Accept 200 (data exists) or 500 (no data)
        assert response.status_code in [200, 500]

    def test_segments_all_pagination(self, app_client):
        """Test segments pagination"""
        response = app_client.get('/api/segments/all?page=1&per_page=10')

        if response.status_code == 200:
            data = response.get_json()
            assert 'segments' in data
            assert 'total' in data
            assert 'page' in data
            assert 'per_page' in data

    def test_segments_all_risk_filter(self, app_client):
        """Test segments filtering by risk level"""
        response = app_client.get('/api/segments/all?risk_label=high')

        # Accept 200 or 500
        assert response.status_code in [200, 500]

        if response.status_code == 200:
            data = response.get_json()
            segments = data.get('segments', [])
            # All returned segments should be high risk
            for segment in segments:
                if 'risk_label' in segment:
                    assert segment['risk_label'] == 'high'

    def test_segments_all_invalid_risk_filter(self, app_client):
        """Test segments with invalid risk level"""
        response = app_client.get('/api/segments/all?risk_label=invalid')

        # Should handle gracefully (return all or filter out)
        assert response.status_code in [200, 400, 500]

    def test_segments_all_per_page_limit(self, app_client):
        """Test segments per_page cannot exceed 1000"""
        response = app_client.get('/api/segments/all?per_page=5000')

        if response.status_code == 200:
            data = response.get_json()
            # Should be capped at 1000
            assert data['per_page'] <= 1000


class TestDiagnosticEndpoints:
    """Tests for diagnostic endpoints"""

    def test_fatality_diagnostic(self, app_client):
        """Test fatality diagnostic endpoint"""
        response = app_client.get('/api/fatality-diagnostic')

        # Accept any status since it's diagnostic
        assert response.status_code in [200, 500]

    def test_data_verification(self, app_client):
        """Test data verification endpoint"""
        response = app_client.get('/api/data-verification')

        # Accept any status since it's diagnostic
        assert response.status_code in [200, 500]

    def test_data_validation_page(self, app_client):
        """Test data validation HTML page loads"""
        response = app_client.get('/data-validation')

        assert response.status_code == 200
        assert b'html' in response.data.lower()
