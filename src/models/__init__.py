"""
Models package for Toronto Road Segment Crash Risk Prediction

This package contains machine learning models and evaluation tools.
ModelEvaluator is not imported here to avoid pulling in matplotlib (NumPy 2.x
incompatibility). Import directly: from src.models.model_evaluator import ModelEvaluator
"""

from .legacy_classifier import ModelTrainer
from .temporal_count import TemporalCountModelTrainer
from .hurdle import HurdleTemporalTrainer

__all__ = [
    'ModelTrainer',
    'TemporalCountModelTrainer',
    'HurdleTemporalTrainer',
]