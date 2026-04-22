"""
Evaluation Metrics

Comprehensive metrics for model evaluation.

Author: Alireza Pourmoslemi
Email: apmath99@gmail.com
"""

import numpy as np
from typing import Dict, Any

try:
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
        average_precision_score,
        confusion_matrix,
        classification_report
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available for metrics")


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray = None
) -> Dict[str, float]:
    """
    Compute all evaluation metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (optional)
    
    Returns:
        Dictionary of metrics
    """
    if not SKLEARN_AVAILABLE:
        return _compute_basic_metrics(y_true, y_pred)
    
    # Basic binary metrics
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, average='binary', zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, average='binary', zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, average='binary', zero_division=0))
    }
    
    # Add macro and weighted F1 (important for imbalanced data)
    metrics['f1_macro'] = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
    metrics['f1_weighted'] = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
    
    # Add macro precision and recall
    metrics['precision_macro'] = float(precision_score(y_true, y_pred, average='macro', zero_division=0))
    metrics['recall_macro'] = float(recall_score(y_true, y_pred, average='macro', zero_division=0))
    
    # Add ROC-AUC if probabilities provided
    if y_proba is not None:
        try:
            if y_proba.ndim == 2 and y_proba.shape[1] == 2:
                metrics['roc_auc'] = float(roc_auc_score(y_true, y_proba[:, 1]))
                # Add PR-AUC (more informative for imbalanced data)
                metrics['pr_auc'] = float(average_precision_score(y_true, y_proba[:, 1]))
            else:
                metrics['roc_auc'] = float(roc_auc_score(y_true, y_proba))
                metrics['pr_auc'] = float(average_precision_score(y_true, y_proba))
        except:
            pass
    
    # Add confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm.tolist()
    
    if cm.shape == (2, 2):
        metrics['tn'] = int(cm[0, 0])
        metrics['fp'] = int(cm[0, 1])
        metrics['fn'] = int(cm[1, 0])
        metrics['tp'] = int(cm[1, 1])
    
    return metrics


def _compute_basic_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
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
        'tn': int(tn)
    }


def print_metrics(metrics: Dict[str, Any], title: str = "Metrics"):
    """
    Print metrics in formatted way.
    
    Args:
        metrics: Dictionary of metrics
        title: Title to print
    """
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    
    # Print main metrics
    for key in ['accuracy', 'precision', 'recall', 'f1', 'f1_macro', 'f1_weighted', 'roc_auc', 'pr_auc']:
        if key in metrics:
            print(f"{key.upper():12s}: {metrics[key]:.4f}")
    
    # Print confusion matrix if available
    if 'tp' in metrics:
        print(f"\nConfusion Matrix:")
        print(f"  TN: {metrics.get('tn', 0):4d}  FP: {metrics.get('fp', 0):4d}")
        print(f"  FN: {metrics.get('fn', 0):4d}  TP: {metrics.get('tp', 0):4d}")
    
    print(f"{'='*60}\n")


def compare_models(results: list, metric: str = 'f1') -> Dict[str, Any]:
    """
    Compare multiple model results.
    
    Args:
        results: List of result dictionaries
        metric: Metric to compare
    
    Returns:
        Comparison summary
    """
    comparison = {
        'best_model': None,
        'best_score': 0.0,
        'rankings': []
    }
    
    # Sort by metric
    sorted_results = sorted(
        results,
        key=lambda x: x.get(metric, 0),
        reverse=True
    )
    
    if sorted_results:
        comparison['best_model'] = sorted_results[0].get('model', 'unknown')
        comparison['best_score'] = sorted_results[0].get(metric, 0)
        
        comparison['rankings'] = [
            {
                'rank': i + 1,
                'model': r.get('model', 'unknown'),
                'score': r.get(metric, 0)
            }
            for i, r in enumerate(sorted_results)
        ]
    
    return comparison
