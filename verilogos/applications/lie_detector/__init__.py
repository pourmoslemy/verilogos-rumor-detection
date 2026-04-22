"""
Topological Lie Detector

A misinformation detection system based on topological truth geometry.
Uses persistent homology and simplicial complex analysis to detect fake news
by analyzing the topological structure of narrative graphs.

Author: Alireza Pourmoslemi
Email: apmath99@gmail.com

Pipeline:
    Text -> Graph -> Simplicial Complex -> Persistent Homology -> 
    TruthShape -> ML Classifier -> Explanation

Components:
    - core: Truth geometry and shape encoding
    - data: Dataset loading and preprocessing
    - topology: Complex builders for text and networks
    - features: Persistence and topology feature extraction
    - models: ML classifiers and training
    - evaluation: Metrics, visualization, explainability
"""

from verilogos.applications.lie_detector.pipeline import LieDetectorPipeline
from verilogos.applications.lie_detector.core.truth_geometry import TruthGeometry
from verilogos.applications.lie_detector.core.shape_encoder import ShapeEncoder

__version__ = "0.1.0"
__author__ = "Alireza Pourmoslemi"
__email__ = "apmath99@gmail.com"

__all__ = [
    "LieDetectorPipeline",
    "TruthGeometry",
    "ShapeEncoder",
]
