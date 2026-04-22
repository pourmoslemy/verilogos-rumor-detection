"""
Classical Models - Baseline classifiers

Implements Logistic Regression, Random Forest, SVM, and XGBoost.

Author: Alireza Pourmoslemi
Email: apmath99@gmail.com
"""

import numpy as np
from typing import Dict, Any

from verilogos.experiments.models.base import BaseModel

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


class LogisticRegressionModel(BaseModel):
    """
    Logistic Regression classifier.
    
    Example:
        >>> model = LogisticRegressionModel()
        >>> model.fit(X_train, y_train)
        >>> metrics = model.evaluate(X_test, y_test)
    """
    
    def __init__(self, **kwargs):
        """Initialize Logistic Regression."""
        super().__init__(name="LogisticRegression")
        
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required for LogisticRegression")
        
        self.model = LogisticRegression(
            max_iter=1000,
            class_weight='balanced',  # FIXED: Handle class imbalance
            random_state=42,
            **kwargs
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs):
        """Train the model."""
        self.model.fit(X, y)
        self.is_trained = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        return self.model.predict_proba(X)


class RandomForestModel(BaseModel):
    """
    Random Forest classifier.
    
    Example:
        >>> model = RandomForestModel(n_estimators=100)
        >>> model.fit(X_train, y_train)
        >>> metrics = model.evaluate(X_test, y_test)
    """
    
    def __init__(self, n_estimators: int = 100, **kwargs):
        """Initialize Random Forest."""
        super().__init__(name="RandomForest")
        
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required for RandomForest")
        
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            class_weight='balanced',  # FIXED: Handle class imbalance
            random_state=42,
            n_jobs=-1,
            **kwargs
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs):
        """Train the model."""
        self.model.fit(X, y)
        self.is_trained = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        return self.model.predict_proba(X)
    
    def get_feature_importance(self) -> np.ndarray:
        """Get feature importances."""
        if not self.is_trained:
            raise ValueError("Model not trained")
        return self.model.feature_importances_


class SVMModel(BaseModel):
    """
    Support Vector Machine classifier.
    
    Example:
        >>> model = SVMModel(kernel='rbf')
        >>> model.fit(X_train, y_train)
        >>> metrics = model.evaluate(X_test, y_test)
    """
    
    def __init__(self, kernel: str = 'rbf', **kwargs):
        """Initialize SVM."""
        super().__init__(name="SVM")
        
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required for SVM")
        
        self.model = SVC(
            kernel=kernel,
            probability=True,
            class_weight='balanced',  # FIXED: Handle class imbalance
            random_state=42,
            **kwargs
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs):
        """Train the model."""
        self.model.fit(X, y)
        self.is_trained = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        return self.model.predict_proba(X)


class XGBoostModel(BaseModel):
    """
    XGBoost classifier.
    
    Example:
        >>> model = XGBoostModel(n_estimators=100)
        >>> model.fit(X_train, y_train)
        >>> metrics = model.evaluate(X_test, y_test)
    """
    
    def __init__(self, n_estimators: int = 100, **kwargs):
        """Initialize XGBoost."""
        super().__init__(name="XGBoost")
        
        if not XGBOOST_AVAILABLE:
            raise ImportError("xgboost required for XGBoostModel")
        
        # Calculate scale_pos_weight for imbalance (will be set during fit)
        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss',
            **kwargs
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs):
        """Train the model with automatic class weight balancing."""
        # Calculate scale_pos_weight for XGBoost (handles imbalance)
        unique, counts = np.unique(y, return_counts=True)
        if len(unique) == 2:
            neg_count = counts[0] if unique[0] == 0 else counts[1]
            pos_count = counts[1] if unique[1] == 1 else counts[0]
            scale_pos_weight = neg_count / pos_count
            self.model.set_params(scale_pos_weight=scale_pos_weight)
            print(f"XGBoost: Set scale_pos_weight={scale_pos_weight:.2f} for class imbalance")
        
        self.model.fit(X, y)
        self.is_trained = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        return self.model.predict_proba(X)
    
    def get_feature_importance(self) -> np.ndarray:
        """Get feature importances."""
        if not self.is_trained:
            raise ValueError("Model not trained")
        return self.model.feature_importances_


def get_classical_model(model_name: str, **kwargs) -> BaseModel:
    """
    Factory function to get classical model by name.
    
    Args:
        model_name: Name of model ('lr', 'rf', 'svm', 'xgboost')
        **kwargs: Model parameters
    
    Returns:
        Model instance
    """
    model_name = model_name.lower()
    
    if model_name in ['lr', 'logistic', 'logisticregression']:
        return LogisticRegressionModel(**kwargs)
    elif model_name in ['rf', 'randomforest']:
        return RandomForestModel(**kwargs)
    elif model_name in ['svm', 'svc']:
        return SVMModel(**kwargs)
    elif model_name in ['xgb', 'xgboost']:
        return XGBoostModel(**kwargs)
    else:
        raise ValueError(f"Unknown model: {model_name}")
