"""
Models module for machine learning classifiers.
"""

from verilogos.applications.lie_detector.models.classifier import LieDetectorClassifier
from verilogos.applications.lie_detector.models.training import ModelTrainer

__all__ = [
    'LieDetectorClassifier',
    'ModelTrainer',
]
