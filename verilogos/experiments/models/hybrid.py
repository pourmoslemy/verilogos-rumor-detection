"""
Hybrid Model - Combines topology and text features

Author: Alireza Pourmoslemi
Email: apmath99@gmail.com
"""

import numpy as np
from typing import Dict, Any

from verilogos.experiments.models.base import BaseModel

try:
    from sklearn.ensemble import RandomForestClassifier
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class HybridModel(BaseModel):
    """
    Hybrid model combining topology and text features.
    
    Concatenates topological features with text features (TF-IDF or embeddings)
    and trains a unified classifier.
    
    Example:
        >>> model = HybridModel()
        >>> X_combined = np.hstack([X_topology, X_text])
        >>> model.fit(X_combined, y_train)
        >>> metrics = model.evaluate(X_combined_test, y_test)
    """
    
    def __init__(self, n_estimators: int = 100, **kwargs):
        """
        Initialize hybrid model.
        
        Args:
            n_estimators: Number of trees
            **kwargs: Additional parameters
        """
        super().__init__(name="Hybrid")
        
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required")
        
        self.classifier = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=42,
            n_jobs=-1,
            **kwargs
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs):
        """
        Train hybrid model.
        
        Args:
            X: Combined feature matrix (topology + text)
            y: Labels
            **kwargs: Additional parameters
        """
        self.classifier.fit(X, y)
        self.is_trained = True
        
        print(f"Trained {self.name} on {X.shape[0]} samples with {X.shape[1]} features")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        return self.classifier.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        return self.classifier.predict_proba(X)
    
    def get_feature_importance(self) -> np.ndarray:
        """Get feature importances."""
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        return self.classifier.feature_importances_


class TopologyTextEnsemble(BaseModel):
    """
    Ensemble that trains separate models on topology and text features,
    then combines predictions.
    """
    
    def __init__(
        self,
        topology_weight: float = 0.5,
        text_weight: float = 0.5,
        **kwargs
    ):
        """
        Initialize ensemble.
        
        Args:
            topology_weight: Weight for topology model
            text_weight: Weight for text model
            **kwargs: Model parameters
        """
        super().__init__(name="TopologyTextEnsemble")
        
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required")
        
        self.topology_weight = topology_weight
        self.text_weight = text_weight
        
        self.topology_model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
        
        self.text_model = RandomForestClassifier(
            n_estimators=100,
            random_state=43,
            n_jobs=-1
        )
        
        self.topology_dim = None
    
    def fit(
        self,
        X_topology: np.ndarray,
        X_text: np.ndarray,
        y: np.ndarray,
        **kwargs
    ):
        """
        Train both models separately.
        
        Args:
            X_topology: Topology features
            X_text: Text features
            y: Labels
            **kwargs: Additional parameters
        """
        self.topology_dim = X_topology.shape[1]
        
        print("Training topology model...")
        self.topology_model.fit(X_topology, y)
        
        print("Training text model...")
        self.text_model.fit(X_text, y)
        
        self.is_trained = True
        
        print(f"Trained ensemble: {X_topology.shape[1]} topology features, "
              f"{X_text.shape[1]} text features")
    
    def predict(self, X_topology: np.ndarray, X_text: np.ndarray) -> np.ndarray:
        """
        Predict using weighted ensemble.
        
        Args:
            X_topology: Topology features
            X_text: Text features
        
        Returns:
            Predictions
        """
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        # Get probabilities from both models
        proba_topology = self.topology_model.predict_proba(X_topology)
        proba_text = self.text_model.predict_proba(X_text)
        
        # Weighted combination
        proba_combined = (
            self.topology_weight * proba_topology +
            self.text_weight * proba_text
        )
        
        # Return class with highest probability
        return np.argmax(proba_combined, axis=1)
    
    def predict_proba(
        self,
        X_topology: np.ndarray,
        X_text: np.ndarray
    ) -> np.ndarray:
        """
        Predict probabilities using weighted ensemble.
        
        Args:
            X_topology: Topology features
            X_text: Text features
        
        Returns:
            Probabilities
        """
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        proba_topology = self.topology_model.predict_proba(X_topology)
        proba_text = self.text_model.predict_proba(X_text)
        
        return (
            self.topology_weight * proba_topology +
            self.text_weight * proba_text
        )
