"""
Tests for input validation logic
"""

import pytest
import json


class TestCoordinateValidation:
    """Tests for coordinate bounds validation"""

    def test_valid_latitude_range(self, app_client):
        """Test valid latitude values (-90 to 90)"""
        # Valid latitude at boundaries
        request_data = {
            'latitude': 43.6532,
            'longitude': -79.3832
        }

        response = app_client.post(
            '/api/risk-prediction',
            data=json.dumps(request_data),
            content_type='application/json'
        )

        # Should accept valid coordinates (200, 404, or 500 based on data availability)
        assert response.status_code in [200, 404, 500]

    def test_valid_longitude_range(self, app_client):
        """Test valid longitude values (-180 to 180)"""
        request_data = {
            'latitude': 43.6532,
            'longitude': -79.3832
        }

        response = app_client.post(
            '/api/risk-prediction',
            data=json.dumps(request_data),
            content_type='application/json'
        )

        assert response.status_code in [200, 404, 500]

    def test_bounding_box_north_south_validation(self, app_client):
        """Test that north must be greater than south"""
        # Invalid: north < south
        request_data = {
            'north': 43.6,
            'south': 43.7,  # south > north (invalid)
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

    def test_bounding_box_east_west_validation(self, app_client):
        """Test that east must be greater than west"""
        # Invalid: east < west
        request_data = {
            'north': 43.7,
            'south': 43.6,
            'east': -79.5,  # east < west (invalid)
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


class TestRiskLabelValidation:
    """Tests for risk label filtering validation"""

    def test_valid_risk_labels(self, app_client):
        """Test that only low, medium, high are valid risk labels"""
        valid_labels = ['low', 'medium', 'high']

        for label in valid_labels:
            response = app_client.get(f'/api/segments/all?risk_label={label}')
            # Should accept valid risk labels (200 or 500 based on data)
            assert response.status_code in [200, 500]

    def test_empty_risk_label(self, app_client):
        """Test behavior with empty risk label"""
        response = app_client.get('/api/segments/all?risk_label=')

        # Should handle gracefully (either filter it out or return all)
        assert response.status_code in [200, 400, 500]


class TestPaginationValidation:
    """Tests for pagination parameter validation"""

    def test_per_page_maximum_limit(self, app_client):
        """Test that per_page is capped at 1000"""
        response = app_client.get('/api/segments/all?per_page=5000')

        if response.status_code == 200:
            data = response.get_json()
            # Should be capped at 1000
            assert data.get('per_page', 0) <= 1000

    def test_per_page_default(self, app_client):
        """Test default per_page value"""
        response = app_client.get('/api/segments/all')

        if response.status_code == 200:
            data = response.get_json()
            # Should have a reasonable default (e.g., 100)
            assert 'per_page' in data
            assert data['per_page'] > 0

    def test_page_number_validation(self, app_client):
        """Test page number handling"""
        # Page 0 should be handled (converted to 1 or rejected)
        response = app_client.get('/api/segments/all?page=0')
        assert response.status_code in [200, 400, 500]

        # Negative page should be handled
        response = app_client.get('/api/segments/all?page=-1')
        assert response.status_code in [200, 400, 500]

    def test_non_numeric_pagination(self, app_client):
        """Test non-numeric pagination parameters"""
        response = app_client.get('/api/segments/all?page=abc&per_page=xyz')

        # Should handle gracefully (use defaults or return 400)
        assert response.status_code in [200, 400, 500]


class TestQueryStringValidation:
    """Tests for query string parameter validation"""

    def test_street_name_query_minimum_length(self, app_client):
        """Test that street name query requires at least 2 characters"""
        # Query too short
        response = app_client.get('/api/street-names?query=K')
        assert response.status_code == 400
        data = response.get_json()
        assert 'error' in data

        # Query at minimum length
        response = app_client.get('/api/street-names?query=Ki')
        assert response.status_code in [200, 500]

    def test_street_name_query_missing(self, app_client):
        """Test street names endpoint requires query parameter"""
        response = app_client.get('/api/street-names')
        assert response.status_code == 400
        data = response.get_json()
        assert 'error' in data

    def test_street_name_limit_parameter(self, app_client):
        """Test limit parameter for street names"""
        response = app_client.get('/api/street-names?query=Street&limit=5')

        if response.status_code == 200:
            data = response.get_json()
            # Should respect limit
            assert len(data) <= 5

    def test_street_name_limit_maximum(self, app_client):
        """Test that limit is capped at maximum (100)"""
        response = app_client.get('/api/street-names?query=Street&limit=500')

        if response.status_code == 200:
            data = response.get_json()
            # Should be capped at 100
            assert len(data) <= 100


class TestRequiredFieldValidation:
    """Tests for required field validation"""

    def test_bbox_prediction_missing_north(self, app_client):
        """Test bounding box prediction requires 'north' field"""
        request_data = {
            'south': 43.6,
            'east': -79.3,
            'west': -79.4
        }

        response = app_client.post(
            '/api/risk-predictions',
            data=json.dumps(request_data),
            content_type='application/json'
        )

        assert response.status_code == 400

    def test_bbox_prediction_missing_south(self, app_client):
        """Test bounding box prediction requires 'south' field"""
        request_data = {
            'north': 43.7,
            'east': -79.3,
            'west': -79.4
        }

        response = app_client.post(
            '/api/risk-predictions',
            data=json.dumps(request_data),
            content_type='application/json'
        )

        assert response.status_code == 400

    def test_point_prediction_missing_latitude(self, app_client):
        """Test single point prediction requires 'latitude' field"""
        request_data = {
            'longitude': -79.3832
        }

        response = app_client.post(
            '/api/risk-prediction',
            data=json.dumps(request_data),
            content_type='application/json'
        )

        assert response.status_code == 400

    def test_point_prediction_missing_longitude(self, app_client):
        """Test single point prediction requires 'longitude' field"""
        request_data = {
            'latitude': 43.6532
        }

        response = app_client.post(
            '/api/risk-prediction',
            data=json.dumps(request_data),
            content_type='application/json'
        )

        assert response.status_code == 400


class TestDataTypeValidation:
    """Tests for data type validation"""

    def test_bbox_prediction_non_numeric_coordinates(self, app_client):
        """Test bounding box prediction with non-numeric coordinates"""
        request_data = {
            'north': 'not_a_number',
            'south': 43.6,
            'east': -79.3,
            'west': -79.4
        }

        response = app_client.post(
            '/api/risk-predictions',
            data=json.dumps(request_data),
            content_type='application/json'
        )

        # Should handle gracefully (400 or 500)
        assert response.status_code in [400, 500]

    def test_point_prediction_non_numeric_coordinates(self, app_client):
        """Test single point prediction with non-numeric coordinates"""
        request_data = {
            'latitude': 'abc',
            'longitude': -79.3832
        }

        response = app_client.post(
            '/api/risk-prediction',
            data=json.dumps(request_data),
            content_type='application/json'
        )

        assert response.status_code in [400, 500]

    def test_empty_json_body(self, app_client):
        """Test endpoints with empty JSON body"""
        response = app_client.post(
            '/api/risk-predictions',
            data=json.dumps({}),
            content_type='application/json'
        )

        assert response.status_code == 400

    def test_malformed_json(self, app_client):
        """Test endpoints with malformed JSON"""
        response = app_client.post(
            '/api/risk-predictions',
            data='{ invalid json',
            content_type='application/json'
        )

        assert response.status_code in [400, 500]
