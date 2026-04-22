"""
Data module for loading and preprocessing fake news datasets.

Supports FakeNewsNet (Politifact) and custom datasets.
"""

from verilogos.applications.lie_detector.data.dataset import FakeNewsDataset
from verilogos.applications.lie_detector.data.loader import DataLoader
from verilogos.applications.lie_detector.data.preprocessor import TextPreprocessor

__all__ = [
    'FakeNewsDataset',
    'DataLoader',
    'TextPreprocessor',
]
