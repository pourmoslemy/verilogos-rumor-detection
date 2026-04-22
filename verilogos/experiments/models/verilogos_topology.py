"""
VeriLogos Topology Model

Pure topology-based model using Betti numbers and persistence features.

Author: Alireza Pourmoslemi
Email: apmath99@gmail.com
"""

import numpy as np
from typing import Dict, Any, List

from verilogos.experiments.models.base import BaseModel

try:
    from sklearn.ensemble import RandomForestClassifier
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class VeriLogosTopologyModel(BaseModel):
    """
    Pure topology-based model using VeriLogos core features.
    
    Extracts topological features (Betti numbers, persistence) and trains
    a classifier on these features alone.
    
    Example:
        >>> model = VeriLogosTopologyModel()
        >>> model.fit(X_topology, y_train)
        >>> metrics = model.evaluate(X_topology_test, y_test)
    """
    
    def __init__(self, n_estimators: int = 100, **kwargs):
        """
        Initialize VeriLogos topology model.
        
        Args:
            n_estimators: Number of trees in RF classifier
            **kwargs: Additional classifier parameters
        """
        super().__init__(name="VeriLogosTopology")
        
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required")
        
        self.classifier = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=42,
            n_jobs=-1,
            **kwargs
        )
        
        self.feature_names = []
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs):
        """
        Train the model on topology features.
        
        Args:
            X: Topology feature matrix (n_samples, n_features)
            y: Labels (n_samples,)
            **kwargs: Additional training parameters
        """
        # Validate input
        if X.shape[0] != len(y):
            raise ValueError(f"X and y size mismatch: {X.shape[0]} vs {len(y)}")
        
        # Train classifier
        self.classifier.fit(X, y)
        self.is_trained = True
        
        print(f"Trained {self.name} on {X.shape[0]} samples with {X.shape[1]} features")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Topology feature matrix
        
        Returns:
            Predictions
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call fit() first.")
        
        return self.classifier.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Topology feature matrix
        
        Returns:
            Probabilities (n_samples, 2)
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call fit() first.")
        
        return self.classifier.predict_proba(X)
    
    def get_feature_importance(self) -> np.ndarray:
        """
        Get feature importances.
        
        Returns:
            Feature importance scores
        """
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        return self.classifier.feature_importances_
    
    def get_top_features(self, n: int = 10) -> List[tuple]:
        """
        Get top N most important features.
        
        Args:
            n: Number of top features
        
        Returns:
            List of (feature_name, importance) tuples
        """
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        importances = self.classifier.feature_importances_
        
        if self.feature_names:
            features = self.feature_names
        else:
            features = [f"feature_{i}" for i in range(len(importances))]
        
        # Sort by importance
        sorted_idx = np.argsort(importances)[::-1][:n]
        
        return [(features[i], importances[i]) for i in sorted_idx]
    
    def set_feature_names(self, names: List[str]):
        """
        Set feature names for interpretability.
        
        Args:
            names: List of feature names
        """
        self.feature_names = names


class VeriLogosTopologyEnsemble(BaseModel):
    """
    Ensemble of topology models with different configurations.
    
    Combines multiple topology-based classifiers for improved performance.
    """
    
    def __init__(self, n_models: int = 3, **kwargs):
        """
        Initialize ensemble.
        
        Args:
            n_models: Number of models in ensemble
            **kwargs: Model parameters
        """
        super().__init__(name="VeriLogosTopologyEnsemble")
        
        self.models = [
            VeriLogosTopologyModel(n_estimators=100, max_depth=None),
            VeriLogosTopologyModel(n_estimators=150, max_depth=20),
            VeriLogosTopologyModel(n_estimators=200, max_depth=15)
        ][:n_models]
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs):
        """Train all models in ensemble."""
        for i, model in enumerate(self.models):
            print(f"Training ensemble model {i+1}/{len(self.models)}")
            model.fit(X, y)
        
        self.is_trained = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using majority voting."""
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        # Get predictions from all models
        predictions = np.array([model.predict(X) for model in self.models])
        
        # Majority vote
        return np.round(predictions.mean(axis=0)).astype(int)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities using averaging."""
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        # Get probabilities from all models
        probas = np.array([model.predict_proba(X) for model in self.models])
        
        # Average probabilities
        return probas.mean(axis=0)
