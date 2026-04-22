"""
Visualization Module

Plotting and visualization for persistence diagrams, barcodes, and results.

Author: Alireza Pourmoslemi
Email: apmath99@gmail.com
"""

import numpy as np
from typing import Dict, List, Tuple, Optional

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Install with: pip install matplotlib")


class Visualizer:
    """
    Visualization utilities for lie detector.
    
    Example:
        >>> viz = Visualizer()
        >>> viz.plot_persistence_diagram(barcodes)
        >>> viz.save_figure("persistence.png")
    """
    
    def __init__(self, figsize: Tuple[int, int] = (10, 6)):
        """
        Initialize visualizer.
        
        Args:
            figsize: Default figure size
        """
        self.figsize = figsize
        
        if not MATPLOTLIB_AVAILABLE:
            print("Matplotlib not available. Visualization disabled.")
    
    def plot_persistence_diagram(
        self,
        barcodes: Dict[int, List[Tuple[float, Optional[float]]]],
        title: str = "Persistence Diagram"
    ):
        """Plot persistence diagram."""
        if not MATPLOTLIB_AVAILABLE:
            return
        
        plt.figure(figsize=self.figsize)
        
        colors = ['blue', 'red', 'green']
        labels = ['H0', 'H1', 'H2']
        
        for dim, intervals in barcodes.items():
            if dim >= len(colors):
                continue
            
            births = []
            deaths = []
            
            for birth, death in intervals:
                if death is not None:
                    births.append(birth)
                    deaths.append(death)
            
            if births:
                plt.scatter(births, deaths, c=colors[dim], label=labels[dim], alpha=0.6)
        
        # Diagonal line
        max_val = max([d for intervals in barcodes.values() 
                      for b, d in intervals if d is not None] + [1])
        plt.plot([0, max_val], [0, max_val], 'k--', alpha=0.3)
        
        plt.xlabel('Birth')
        plt.ylabel('Death')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    def plot_barcode(
        self,
        barcodes: Dict[int, List[Tuple[float, Optional[float]]]],
        title: str = "Persistence Barcode"
    ):
        """Plot persistence barcode."""
        if not MATPLOTLIB_AVAILABLE:
            return
        
        plt.figure(figsize=self.figsize)
        
        y_pos = 0
        colors = ['blue', 'red', 'green']
        
        for dim in sorted(barcodes.keys()):
            intervals = barcodes[dim]
            color = colors[dim] if dim < len(colors) else 'gray'
            
            for birth, death in intervals:
                if death is None:
                    death = birth + 10  # Represent infinity
                
                plt.plot([birth, death], [y_pos, y_pos], color=color, linewidth=2)
                y_pos += 1
        
        plt.xlabel('Time')
        plt.ylabel('Features')
        plt.title(title)
        plt.grid(True, alpha=0.3)
    
    def plot_confusion_matrix(self, cm: np.ndarray, title: str = "Confusion Matrix"):
        """Plot confusion matrix."""
        if not MATPLOTLIB_AVAILABLE:
            return
        
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(title)
        plt.colorbar()
        
        labels = ['Fake', 'Real']
        tick_marks = np.arange(len(labels))
        plt.xticks(tick_marks, labels)
        plt.yticks(tick_marks, labels)
        
        # Add text annotations
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, str(cm[i, j]),
                        ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
    
    def save_figure(self, filepath: str):
        """Save current figure."""
        if MATPLOTLIB_AVAILABLE:
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {filepath}")
