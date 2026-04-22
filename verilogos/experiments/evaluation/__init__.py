"""
Evaluation Module

Metrics and visualization for experiments.

Author: Alireza Pourmoslemi
Email: apmath99@gmail.com
"""

from verilogos.experiments.evaluation.metrics import (
    compute_metrics,
    print_metrics,
    compare_models
)

from verilogos.experiments.evaluation.visualization import (
    plot_confusion_matrix,
    plot_roc_curve
)

__all__ = [
    'compute_metrics',
    'print_metrics',
    'compare_models',
    'plot_confusion_matrix',
    'plot_roc_curve',
]
