"""
Evaluation Metrics Module

Comprehensive evaluation metrics for fake news detection.

Author: Alireza Pourmoslemi
Email: apmath99@gmail.com
"""

import numpy as np
from typing import Dict, List, Tuple

try:
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        confusion_matrix, classification_report, roc_auc_score, roc_curve
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class EvaluationMetrics:
    """
    Computes evaluation metrics for classification.
    
    Example:
        >>> metrics = EvaluationMetrics()
        >>> results = metrics.compute_all_metrics(y_true, y_pred, y_proba)
    """
    
    def compute_all_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray = None
    ) -> Dict[str, any]:
        """
        Compute all metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (optional)
            
        Returns:
            Dictionary of metrics
        """
        if not SKLEARN_AVAILABLE:
            return self._compute_basic_metrics(y_true, y_pred)
        
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average='binary')
        metrics['recall'] = recall_score(y_true, y_pred, average='binary')
        metrics['f1'] = f1_score(y_true, y_pred, average='binary')
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm
        metrics['tn'] = int(cm[0, 0])
        metrics['fp'] = int(cm[0, 1])
        metrics['fn'] = int(cm[1, 0])
        metrics['tp'] = int(cm[1, 1])
        
        # ROC AUC if probabilities provided
        if y_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
            fpr, tpr, thresholds = roc_curve(y_true, y_proba)
            metrics['roc_curve'] = (fpr, tpr, thresholds)
        
        return metrics
    
    def _compute_basic_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Compute basic metrics without sklearn."""
        correct = np.sum(y_true == y_pred)
        total = len(y_true)
        
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'accuracy': correct / total,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn),
            'tn': int(tn),
        }
    
    def print_report(self, metrics: Dict[str, any]):
        """Print formatted metrics report."""
        print("\n" + "="*50)
        print("EVALUATION METRICS")
        print("="*50)
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1 Score:  {metrics['f1']:.4f}")
        
        if 'roc_auc' in metrics:
            print(f"ROC AUC:   {metrics['roc_auc']:.4f}")
        
        print("\nConfusion Matrix:")
        print(f"TN: {metrics['tn']:4d}  FP: {metrics['fp']:4d}")
        print(f"FN: {metrics['fn']:4d}  TP: {metrics['tp']:4d}")
        print("="*50 + "\n")
