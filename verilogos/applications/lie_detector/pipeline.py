"""
Pipeline Module

End-to-end pipeline for topological lie detection.

Author: Alireza Pourmoslemi
Email: apmath99@gmail.com
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from verilogos.applications.lie_detector.data.dataset import FakeNewsDataset
from verilogos.applications.lie_detector.data.preprocessor import TextPreprocessor
from verilogos.applications.lie_detector.topology.text_complex_builder import TextComplexBuilder
from verilogos.applications.lie_detector.core.truth_geometry import TruthGeometry
from verilogos.applications.lie_detector.core.shape_encoder import ShapeEncoder
from verilogos.applications.lie_detector.features.persistence_features import PersistenceFeatureExtractor
from verilogos.applications.lie_detector.models.classifier import LieDetectorClassifier
from verilogos.applications.lie_detector.evaluation.explainability import ExplainabilityEngine


class LieDetectorPipeline:
    """
    Complete end-to-end pipeline for fake news detection.
    
    Pipeline stages:
        1. Text preprocessing
        2. Simplicial complex construction
        3. Persistent homology computation
        4. Feature extraction
        5. Classification
        6. Explanation generation
    
    Example:
        >>> pipeline = LieDetectorPipeline()
        >>> pipeline.train(train_texts, train_labels)
        >>> result = pipeline.predict("Breaking news text...")
    """
    
    def __init__(
        self,
        similarity_threshold: float = 0.3,
        max_dimension: int = 2,
        n_estimators: int = 100,
    ):
        """
        Initialize pipeline.
        
        Args:
            similarity_threshold: Threshold for edge creation in complex
            max_dimension: Maximum homology dimension
            n_estimators: Number of trees in classifier
        """
        self.similarity_threshold = similarity_threshold
        self.max_dimension = max_dimension
        self.n_estimators = n_estimators
        
        # Initialize components
        self.preprocessor = TextPreprocessor()
        self.complex_builder = TextComplexBuilder(
            similarity_threshold=similarity_threshold,
            max_dimension=max_dimension,
        )
        self.truth_geometry = TruthGeometry(max_dimension=max_dimension)
        self.shape_encoder = ShapeEncoder(max_dimension=max_dimension)
        self.feature_extractor = PersistenceFeatureExtractor(max_dimension=max_dimension)
        self.classifier = None
        self.explainer = ExplainabilityEngine()
        
        self.is_trained = False
    
    def train(
        self,
        texts: List[str],
        labels: List[int],
    ):
        """
        Train pipeline on texts and labels.
        
        Args:
            texts: List of article texts
            labels: List of labels (0=real, 1=fake)
        """
        print(f"Training pipeline on {len(texts)} articles...")
        
        # Extract features from all articles
        X_features = []
        y_labels = []
        
        for i, (text, label) in enumerate(zip(texts, labels)):
            if i % 10 == 0:
                print(f"Processing article {i+1}/{len(texts)}")
            
            try:
                # Build complex
                complex = self.complex_builder.build_complex(text)
                
                # Extract features
                features = self.feature_extractor.extract(complex)
                feature_vector = np.array(list(features.values()))
                
                X_features.append(feature_vector)
                y_labels.append(label)
            except Exception as e:
                print(f"Error processing article {i}: {e}")
                continue
        
        X = np.array(X_features)
        y = np.array(y_labels)
        
        print(f"\nFeature matrix shape: {X.shape}")
        print(f"Label distribution: {np.bincount(y)}")
        
        # Train classifier
        self.classifier = LieDetectorClassifier(
            n_estimators=self.n_estimators,
        )
        self.classifier.fit(X, y)
        
        self.is_trained = True
        print("\nTraining complete!")
    
    def predict(self, text: str) -> Dict[str, any]:
        """
        Predict whether text is fake or real.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with prediction, confidence, and explanation
        """
        if not self.is_trained:
            raise ValueError("Pipeline not trained. Call train() first.")
        
        # Build complex
        complex = self.complex_builder.build_complex(text)
        
        # Compute truth shape
        truth_shape = self.truth_geometry.analyze(complex)
        
        # Extract features
        features = self.feature_extractor.extract(complex)
        feature_vector = np.array(list(features.values())).reshape(1, -1)
        
        # Predict
        prediction = self.classifier.predict(feature_vector)[0]
        probabilities = self.classifier.predict_proba(feature_vector)[0]
        confidence = probabilities[prediction]
        
        # Generate explanation
        explanation = self.explainer.explain(
            truth_shape,
            prediction,
            confidence,
        )
        
        return {
            'prediction': int(prediction),
            'label': 'REAL' if prediction == 0 else 'FAKE',
            'confidence': float(confidence),
            'probabilities': {
                'real': float(probabilities[0]),
                'fake': float(probabilities[1]),
            },
            'shape_code': truth_shape.shape_code,
            'truth_shape': truth_shape,
            'explanation': explanation,
        }
    
    def save_model(self, filepath: str):
        """Save trained pipeline."""
        if not self.is_trained:
            raise ValueError("Pipeline not trained. Call train() first.")
        
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self.classifier, f)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained pipeline."""
        import pickle
        with open(filepath, 'rb') as f:
            self.classifier = pickle.load(f)
        self.is_trained = True
        print(f"Model loaded from {filepath}")
