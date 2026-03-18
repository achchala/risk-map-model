"""
Tests for helper functions in app.py
"""

import pytest
import numpy as np
import pandas as pd
from shapely.geometry import LineString, MultiLineString, Point, Polygon
from unittest.mock import Mock, MagicMock
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "backend-api"))


class TestExtractCoordinates:
    """Tests for _extract_coordinates helper function"""

    def test_extract_linestring_coordinates(self):
        """Test extracting coordinates from LineString"""
        from app import _extract_coordinates

        line = LineString([(43.65, -79.38), (43.66, -79.37), (43.67, -79.36)])
        coords = _extract_coordinates(line)

        assert len(coords) == 3
        assert coords[0] == {"latitude": -79.38, "longitude": 43.65}
        assert coords[1] == {"latitude": -79.37, "longitude": 43.66}
        assert coords[2] == {"latitude": -79.36, "longitude": 43.67}

    def test_extract_multilinestring_coordinates(self):
        """Test extracting coordinates from MultiLineString"""
        from app import _extract_coordinates

        line1 = LineString([(43.65, -79.38), (43.66, -79.37)])
        line2 = LineString([(43.67, -79.36), (43.68, -79.35)])
        multi_line = MultiLineString([line1, line2])

        coords = _extract_coordinates(multi_line)

        assert len(coords) == 4
        assert coords[0] == {"latitude": -79.38, "longitude": 43.65}
        assert coords[3] == {"latitude": -79.35, "longitude": 43.68}

    def test_extract_point_coordinates(self):
        """Test extracting coordinates from Point"""
        from app import _extract_coordinates

        point = Point(43.6532, -79.3832)
        coords = _extract_coordinates(point)

        assert len(coords) == 1
        assert coords[0] == {"latitude": -79.3832, "longitude": 43.6532}

    def test_extract_polygon_coordinates(self):
        """Test extracting coordinates from Polygon"""
        from app import _extract_coordinates

        polygon = Polygon([(43.65, -79.38), (43.66, -79.37), (43.67, -79.36), (43.65, -79.38)])
        coords = _extract_coordinates(polygon)

        assert len(coords) > 0
        # Polygon should extract exterior coordinates
        assert all('latitude' in c and 'longitude' in c for c in coords)

    def test_extract_coordinates_empty_geometry(self):
        """Test extracting coordinates from empty/invalid geometry"""
        from app import _extract_coordinates

        # Empty geometry should return empty list
        coords = _extract_coordinates(None)
        assert coords == []

    def test_extract_coordinates_handles_exceptions(self):
        """Test coordinate extraction handles exceptions gracefully"""
        from app import _extract_coordinates

        # Test with mock object that raises exceptions
        mock_geom = Mock()
        mock_geom.coords = Mock(side_effect=Exception("Test error"))
        mock_geom.geoms = None

        coords = _extract_coordinates(mock_geom)
        # Should return empty list on error
        assert coords == []


class TestPredictSegmentRisk:
    """Tests for _predict_segment_risk helper function"""

    def test_predict_segment_risk_success(self, mock_model_trainer, mock_road_segment):
        """Test successful risk prediction"""
        from app import _predict_segment_risk

        # Setup mock trainer with required attributes
        mock_model_trainer.feature_columns = ['total_crashes', 'crash_density']
        mock_model_trainer.label_encoder = Mock()
        mock_model_trainer.label_encoder.classes_ = ['low', 'medium', 'high']
        mock_model_trainer.label_encoder.inverse_transform.return_value = ['high']

        mock_model_trainer.model.predict.return_value = np.array([2])  # high risk encoded
        mock_model_trainer.model.predict_proba.return_value = np.array([[0.1, 0.2, 0.7]])

        risk_label, probabilities, confidence = _predict_segment_risk(
            mock_road_segment, mock_model_trainer
        )

        assert risk_label == 'high'
        assert probabilities['low'] == 0.1
        assert probabilities['medium'] == 0.2
        assert probabilities['high'] == 0.7
        assert confidence == 0.7

    def test_predict_segment_risk_no_feature_columns(self, mock_model_trainer, mock_road_segment):
        """Test prediction when feature_columns is None"""
        from app import _predict_segment_risk

        mock_model_trainer.feature_columns = None

        risk_label, probabilities, confidence = _predict_segment_risk(
            mock_road_segment, mock_model_trainer
        )

        assert risk_label is None
        assert probabilities is None
        assert confidence is None

    def test_predict_segment_risk_missing_features(self, mock_model_trainer, mock_road_segment):
        """Test prediction when segment is missing some features"""
        from app import _predict_segment_risk

        # Setup mock trainer with features not in segment
        mock_model_trainer.feature_columns = ['feature_not_in_segment', 'another_missing']
        mock_model_trainer.label_encoder = Mock()
        mock_model_trainer.label_encoder.classes_ = ['low', 'medium', 'high']
        mock_model_trainer.label_encoder.inverse_transform.return_value = ['low']

        mock_model_trainer.model.predict.return_value = np.array([0])
        mock_model_trainer.model.predict_proba.return_value = np.array([[0.8, 0.15, 0.05]])

        risk_label, probabilities, confidence = _predict_segment_risk(
            mock_road_segment, mock_model_trainer
        )

        # Should default missing features to 0.0 and still predict
        assert risk_label == 'low'
        assert probabilities is not None
        assert confidence == 0.8

    def test_predict_segment_risk_handles_nan_values(self, mock_model_trainer):
        """Test prediction handles NaN values in features"""
        from app import _predict_segment_risk

        # Create segment with NaN values
        segment_with_nan = pd.Series({
            'total_crashes': np.nan,
            'crash_density': 0.5,
            'geometry': Point(43.65, -79.38)
        })

        mock_model_trainer.feature_columns = ['total_crashes', 'crash_density']
        mock_model_trainer.label_encoder = Mock()
        mock_model_trainer.label_encoder.classes_ = ['low', 'medium', 'high']
        mock_model_trainer.label_encoder.inverse_transform.return_value = ['low']

        mock_model_trainer.model.predict.return_value = np.array([0])
        mock_model_trainer.model.predict_proba.return_value = np.array([[0.7, 0.2, 0.1]])

        risk_label, probabilities, confidence = _predict_segment_risk(
            segment_with_nan, mock_model_trainer
        )

        # NaN should be converted to 0.0
        assert risk_label == 'low'
        assert probabilities is not None

    def test_predict_segment_risk_exception_handling(self, mock_road_segment):
        """Test prediction handles exceptions gracefully"""
        from app import _predict_segment_risk

        # Create mock trainer that raises exception
        bad_trainer = Mock()
        bad_trainer.feature_columns = ['col1']
        bad_trainer.model.predict.side_effect = Exception("Model error")

        risk_label, probabilities, confidence = _predict_segment_risk(
            mock_road_segment, bad_trainer
        )

        assert risk_label is None
        assert probabilities is None
        assert confidence is None


class TestConvertToJsonSerializable:
    """Tests for _convert_to_json_serializable helper function"""

    def test_convert_numpy_integer(self):
        """Test converting numpy integer to Python int"""
        from app import _convert_to_json_serializable

        result = _convert_to_json_serializable(np.int64(42))
        assert result == 42
        assert isinstance(result, int)

    def test_convert_numpy_float(self):
        """Test converting numpy float to Python float"""
        from app import _convert_to_json_serializable

        result = _convert_to_json_serializable(np.float64(3.14))
        assert result == 3.14
        assert isinstance(result, float)

    def test_convert_numpy_array(self):
        """Test converting numpy array to list"""
        from app import _convert_to_json_serializable

        arr = np.array([1, 2, 3])
        result = _convert_to_json_serializable(arr)

        assert result == [1, 2, 3]
        assert isinstance(result, list)

    def test_convert_dict_with_numpy_values(self):
        """Test converting dict containing numpy values"""
        from app import _convert_to_json_serializable

        data = {
            'count': np.int64(10),
            'value': np.float32(2.5),
            'array': np.array([1, 2])
        }

        result = _convert_to_json_serializable(data)

        assert result['count'] == 10
        assert isinstance(result['count'], int)
        assert result['value'] == 2.5
        assert isinstance(result['value'], float)
        assert result['array'] == [1, 2]

    def test_convert_list_with_numpy_values(self):
        """Test converting list containing numpy values"""
        from app import _convert_to_json_serializable

        data = [np.int64(1), np.float64(2.5), np.array([3, 4])]
        result = _convert_to_json_serializable(data)

        assert result == [1, 2.5, [3, 4]]
        assert isinstance(result[0], int)
        assert isinstance(result[1], float)

    def test_convert_pandas_na(self):
        """Test converting pandas NA/NaN to None"""
        from app import _convert_to_json_serializable

        result = _convert_to_json_serializable(pd.NA)
        assert result is None

        result = _convert_to_json_serializable(np.nan)
        assert result is None

    def test_convert_nested_structure(self):
        """Test converting deeply nested structure"""
        from app import _convert_to_json_serializable

        data = {
            'level1': {
                'level2': {
                    'count': np.int64(5),
                    'values': [np.float64(1.1), np.float64(2.2)]
                }
            }
        }

        result = _convert_to_json_serializable(data)

        assert result['level1']['level2']['count'] == 5
        assert isinstance(result['level1']['level2']['count'], int)
        assert result['level1']['level2']['values'] == [1.1, 2.2]

    def test_convert_native_types_unchanged(self):
        """Test that native Python types pass through unchanged"""
        from app import _convert_to_json_serializable

        assert _convert_to_json_serializable(42) == 42
        assert _convert_to_json_serializable(3.14) == 3.14
        assert _convert_to_json_serializable("string") == "string"
        assert _convert_to_json_serializable([1, 2, 3]) == [1, 2, 3]
        assert _convert_to_json_serializable({"key": "value"}) == {"key": "value"}
