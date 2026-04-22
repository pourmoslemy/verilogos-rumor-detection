"""
Classifier Module

Machine learning classifiers for fake news detection.

Author: Alireza Pourmoslemi
Email: apmath99@gmail.com
"""

import numpy as np
import pickle
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available. Install with: pip install scikit-learn")


class LieDetectorClassifier:
    """
    Machine learning classifier for fake news detection.
    
    Uses Random Forest with topological features.
    
    Example:
        >>> classifier = LieDetectorClassifier()
        >>> classifier.fit(X_train, y_train)
        >>> predictions = classifier.predict(X_test)
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        random_state: int = 42
    ):
        """
        Initialize classifier.
        
        Args:
            n_estimators: Number of trees in random forest
            max_depth: Maximum tree depth
            random_state: Random seed
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required. Install with: pip install scikit-learn")
        
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
        )
        
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_names = None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Train classifier.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            y: Labels of shape (n_samples,)
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, y)
        
        self.is_fitted = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict labels.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted labels
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Feature matrix
            
        Returns:
            Probability matrix of shape (n_samples, n_classes)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def get_feature_importance(self) -> np.ndarray:
        """
        Get feature importances.
        
        Returns:
            Feature importance array
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        return self.model.feature_importances_
    
    def get_top_features(self, n: int = 10) -> List[Tuple[str, float]]:
        """
        Get top N most important features.
        
        Args:
            n: Number of top features
            
        Returns:
            List of (feature_name, importance) tuples
        """
        importances = self.get_feature_importance()
        
        if self.feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importances))]
        else:
            feature_names = self.feature_names
        
        # Sort by importance
        indices = np.argsort(importances)[::-1][:n]
        
        return [(feature_names[i], importances[i]) for i in indices]
    
    def save(self, filepath: str):
        """
        Save model to file.
        
        Args:
            filepath: Path to save model
        """
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'is_fitted': self.is_fitted,
            'feature_names': self.feature_names,
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'random_state': self.random_state,
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'LieDetectorClassifier':
        """
        Load model from file.
        
        Args:
            filepath: Path to model file
            
        Returns:
            Loaded LieDetectorClassifier
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        classifier = cls(
            n_estimators=model_data['n_estimators'],
            max_depth=model_data['max_depth'],
            random_state=model_data['random_state'],
        )
        
        classifier.model = model_data['model']
        classifier.scaler = model_data['scaler']
        classifier.is_fitted = model_data['is_fitted']
        classifier.feature_names = model_data['feature_names']
        
        return classifier
    
    def set_feature_names(self, names: List[str]):
        """
        Set feature names for interpretability.
        
        Args:
            names: List of feature names
        """
        self.feature_names = names
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            X: Feature matrix
            y: True labels
            
        Returns:
            Dictionary of metrics
        """
        if not SKLEARN_AVAILABLE:
            return {}
        
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        y_pred = self.predict(X)
        
        return {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, average='binary'),
            'recall': recall_score(y, y_pred, average='binary'),
            'f1': f1_score(y, y_pred, average='binary'),
        }
