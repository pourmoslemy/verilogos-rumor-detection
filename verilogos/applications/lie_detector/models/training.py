"""
Training Module

Training loop and optimization for lie detector models.

Author: Alireza Pourmoslemi
Email: apmath99@gmail.com
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path

from verilogos.applications.lie_detector.models.classifier import LieDetectorClassifier


class ModelTrainer:
    """
    Trainer for lie detector models.
    
    Handles training loop, validation, and model selection.
    
    Example:
        >>> trainer = ModelTrainer()
        >>> classifier = trainer.train(X_train, y_train, X_val, y_val)
    """
    
    def __init__(self, output_dir: str = "./models"):
        """
        Initialize trainer.
        
        Args:
            output_dir: Directory to save models
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.best_model = None
        self.best_score = 0.0
        self.training_history = []
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
    ) -> LieDetectorClassifier:
        """
        Train classifier.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            n_estimators: Number of trees
            max_depth: Maximum tree depth
            
        Returns:
            Trained classifier
        """
        print(f"Training with {len(X_train)} samples...")
        
        # Initialize classifier
        classifier = LieDetectorClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
        )
        
        # Train
        classifier.fit(X_train, y_train)
        
        # Evaluate on training set
        train_metrics = classifier.evaluate(X_train, y_train)
        print(f"Training metrics: {train_metrics}")
        
        # Evaluate on validation set if provided
        if X_val is not None and y_val is not None:
            val_metrics = classifier.evaluate(X_val, y_val)
            print(f"Validation metrics: {val_metrics}")
            
            # Track best model
            val_score = val_metrics['f1']
            if val_score > self.best_score:
                self.best_score = val_score
                self.best_model = classifier
                
                # Save best model
                model_path = self.output_dir / "best_model.pkl"
                classifier.save(str(model_path))
        
        # Record history
        self.training_history.append({
            'train_metrics': train_metrics,
            'val_metrics': val_metrics if X_val is not None else None,
        })
        
        return classifier
    
    def hyperparameter_search(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        param_grid: Dict[str, List],
    ) -> Tuple[LieDetectorClassifier, Dict]:
        """
        Perform hyperparameter search.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            param_grid: Dictionary of parameter lists to search
            
        Returns:
            Tuple of (best_classifier, best_params)
        """
        best_classifier = None
        best_params = None
        best_score = 0.0
        
        # Grid search
        n_estimators_list = param_grid.get('n_estimators', [100])
        max_depth_list = param_grid.get('max_depth', [None])
        
        for n_estimators in n_estimators_list:
            for max_depth in max_depth_list:
                print(f"\nTrying n_estimators={n_estimators}, max_depth={max_depth}")
                
                classifier = self.train(
                    X_train, y_train,
                    X_val, y_val,
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                )
                
                val_metrics = classifier.evaluate(X_val, y_val)
                score = val_metrics['f1']
                
                if score > best_score:
                    best_score = score
                    best_classifier = classifier
                    best_params = {
                        'n_estimators': n_estimators,
                        'max_depth': max_depth,
                    }
        
        print(f"\nBest parameters: {best_params}")
        print(f"Best F1 score: {best_score:.4f}")
        
        return best_classifier, best_params
    
    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_folds: int = 5,
    ) -> Dict[str, List[float]]:
        """
        Perform k-fold cross-validation.
        
        Args:
            X: Feature matrix
            y: Labels
            n_folds: Number of folds
            
        Returns:
            Dictionary of metric lists
        """
        from sklearn.model_selection import KFold
        
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        metrics_per_fold = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
        }
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            print(f"\nFold {fold + 1}/{n_folds}")
            
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            classifier = self.train(X_train, y_train)
            metrics = classifier.evaluate(X_val, y_val)
            
            for key, value in metrics.items():
                if key in metrics_per_fold:
                    metrics_per_fold[key].append(value)
        
        # Print average metrics
        print("\nCross-validation results:")
        for key, values in metrics_per_fold.items():
            mean = np.mean(values)
            std = np.std(values)
            print(f"{key}: {mean:.4f} ± {std:.4f}")
        
        return metrics_per_fold
