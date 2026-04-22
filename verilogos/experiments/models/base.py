"""
Base Model - Abstract interface for all models

Author: Alireza Pourmoslemi
Email: apmath99@gmail.com
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List
import numpy as np


class BaseModel(ABC):
    """
    Abstract base class for all models.
    
    All models must implement this interface.
    """
    
    def __init__(self, name: str):
        """
        Initialize model.
        
        Args:
            name: Model name
        """
        self.name = name
        self.is_trained = False
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs):
        """
        Train the model.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Labels (n_samples,)
            **kwargs: Additional training parameters
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Feature matrix (n_samples, n_features)
        
        Returns:
            Predictions (n_samples,)
        """
        pass
    
    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Feature matrix (n_samples, n_features)
        
        Returns:
            Probabilities (n_samples, n_classes)
        """
        pass
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            X: Feature matrix
            y: True labels
        
        Returns:
            Dictionary of metrics
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call fit() first.")
        
        # Make predictions
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)
        
        # Calculate metrics
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        )
        
        metrics = {
            'accuracy': float(accuracy_score(y, y_pred)),
            'precision': float(precision_score(y, y_pred, average='binary', zero_division=0)),
            'recall': float(recall_score(y, y_pred, average='binary', zero_division=0)),
            'f1': float(f1_score(y, y_pred, average='binary', zero_division=0))
        }
        
        # Add ROC-AUC if probabilities available
        if y_proba is not None and y_proba.shape[1] == 2:
            try:
                metrics['roc_auc'] = float(roc_auc_score(y, y_proba[:, 1]))
            except:
                pass
        
        return metrics
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get model parameters.
        
        Returns:
            Dictionary of parameters
        """
        return {'name': self.name}
    
    def set_params(self, **params):
        """
        Set model parameters.
        
        Args:
            **params: Parameters to set
        """
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
