"""
Models package for Toronto Road Segment Crash Risk Prediction

This package contains machine learning models and evaluation tools.
"""

from .model_trainer import ModelTrainer
from .model_evaluator import ModelEvaluator

__all__ = [
    'ModelTrainer',
    'ModelEvaluator'
] 