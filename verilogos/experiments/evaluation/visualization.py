"""
Visualization Module

Visualization utilities for experiment results.

Author: Alireza Pourmoslemi
Email: apmath99@gmail.com
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str] = None,
    save_path: str = None,
    title: str = "Confusion Matrix"
) -> Optional[str]:
    """
    Plot confusion matrix.
    
    Args:
        cm: Confusion matrix
        class_names: Class names
        save_path: Path to save plot
        title: Plot title
    
    Returns:
        Path to saved plot
    """
    if not PLOTTING_AVAILABLE:
        return None
    
    if class_names is None:
        class_names = ['Real', 'Fake']
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax
    )
    
    ax.set_xlabel('Predicted', fontweight='bold')
    ax.set_ylabel('True', fontweight='bold')
    ax.set_title(title, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return save_path
    else:
        plt.show()
        return None


def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auc: float,
    save_path: str = None,
    title: str = "ROC Curve"
) -> Optional[str]:
    """
    Plot ROC curve.
    
    Args:
        fpr: False positive rates
        tpr: True positive rates
        auc: AUC score
        save_path: Path to save
        title: Title
    
    Returns:
        Path to saved plot
    """
    if not PLOTTING_AVAILABLE:
        return None
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(fpr, tpr, linewidth=2, label=f'AUC = {auc:.3f}')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
    
    ax.set_xlabel('False Positive Rate', fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontweight='bold')
    ax.set_title(title, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return save_path
    else:
        plt.show()
        return None
