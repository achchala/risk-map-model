"""
Compatibility shim — re-exports all model classes from their respective submodules.

Individual models live in:
  src/models/legacy_classifier/  — ModelTrainer (RandomForest)
  src/models/temporal_count/     — TemporalCountModelTrainer (HistGBT Poisson)
  src/models/hurdle/             — HurdleTemporalTrainer (two-stage hurdle)
"""

from .legacy_classifier.model_trainer import ModelTrainer
from .temporal_count.model_trainer import TemporalCountModelTrainer
from .hurdle.model_trainer import HurdleTemporalTrainer

__all__ = [
    "ModelTrainer",
    "TemporalCountModelTrainer",
    "HurdleTemporalTrainer",
]
