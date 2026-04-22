"""
Plotting Module - Publication-quality visualizations

Generates paper-ready plots for experiment results.

Author: Alireza Pourmoslemi
Email: apmath99@gmail.com
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Warning: matplotlib/seaborn not available. Plotting disabled.")


class ExperimentPlotter:
    """
    Generate publication-quality plots for experiments.
    
    Features:
    - Model comparison bar charts
    - Ablation performance curves
    - Confusion matrix heatmaps
    - ROC curves
    - Learning curves
    
    Example:
        >>> plotter = ExperimentPlotter(output_dir="results/plots")
        >>> plotter.plot_model_comparison(results, metric="f1")
        >>> plotter.plot_ablation(ablation_results)
    """
    
    def __init__(self, output_dir: str = "plots", style: str = "seaborn-v0_8"):
        """
        Initialize plotter.
        
        Args:
            output_dir: Directory to save plots
            style: Matplotlib style
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if PLOTTING_AVAILABLE:
            # Set style
            try:
                plt.style.use(style)
            except:
                plt.style.use('default')
            
            # Set seaborn defaults
            sns.set_palette("husl")
            
            # Set font sizes for publication
            plt.rcParams.update({
                'font.size': 12,
                'axes.labelsize': 14,
                'axes.titlesize': 16,
                'xtick.labelsize': 11,
                'ytick.labelsize': 11,
                'legend.fontsize': 11,
                'figure.titlesize': 18
            })
    
    def plot_model_comparison(
        self,
        results: List[Dict[str, Any]],
        metric: str = "f1",
        save_name: str = "model_comparison.png",
        title: Optional[str] = None
    ) -> Optional[str]:
        """
        Plot bar chart comparing models.
        
        Args:
            results: List of result dicts with 'model' and metrics
            metric: Metric to plot (accuracy, f1, precision, recall)
            save_name: Output filename
            title: Plot title (optional)
        
        Returns:
            Path to saved plot
        """
        if not PLOTTING_AVAILABLE:
            print("Plotting not available")
            return None
        
        # Extract data
        models = []
        values = []
        errors = []
        
        for result in results:
            if 'model' in result and metric in result:
                models.append(result['model'])
                values.append(result[metric])
                # Add error bars if available
                error_key = f"{metric}_std"
                errors.append(result.get(error_key, 0))
        
        if not models:
            print(f"No data found for metric: {metric}")
            return None
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create bar chart
        x = np.arange(len(models))
        bars = ax.bar(x, values, yerr=errors if any(errors) else None,
                     capsize=5, alpha=0.8, edgecolor='black', linewidth=1.2)
        
        # Color bars by performance
        colors = plt.cm.RdYlGn(np.array(values) / max(values))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        # Customize
        ax.set_xlabel('Model', fontweight='bold')
        ax.set_ylabel(metric.upper(), fontweight='bold')
        ax.set_title(title or f'Model Comparison - {metric.upper()}', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, values)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.3f}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        
        # Save
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved model comparison plot: {save_path}")
        return str(save_path)
    
    def plot_ablation(
        self,
        ablation_results: List[Dict[str, Any]],
        metric: str = "f1",
        save_name: str = "ablation_study.png",
        title: Optional[str] = None
    ) -> Optional[str]:
        """
        Plot ablation study results.
        
        Args:
            ablation_results: List of ablation results
            metric: Metric to plot
            save_name: Output filename
            title: Plot title
        
        Returns:
            Path to saved plot
        """
        if not PLOTTING_AVAILABLE:
            return None
        
        # Extract ablation configs and scores
        configs = []
        scores = []
        
        for result in ablation_results:
            if metric in result:
                # Create config label
                config_parts = []
                for key, value in result.items():
                    if key not in [metric, 'model', 'dataset', 'status', 'duration']:
                        if isinstance(value, bool):
                            if value:
                                config_parts.append(key)
                        else:
                            config_parts.append(f"{key}={value}")
                
                config_label = "+".join(config_parts) if config_parts else "baseline"
                configs.append(config_label)
                scores.append(result[metric])
        
        if not configs:
            print("No ablation data found")
            return None
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot line with markers
        x = np.arange(len(configs))
        ax.plot(x, scores, marker='o', linewidth=2, markersize=8, alpha=0.8)
        
        # Customize
        ax.set_xlabel('Configuration', fontweight='bold')
        ax.set_ylabel(metric.upper(), fontweight='bold')
        ax.set_title(title or f'Ablation Study - {metric.upper()}', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(configs, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Add horizontal line for best score
        best_score = max(scores)
        ax.axhline(y=best_score, color='r', linestyle='--', alpha=0.5, label=f'Best: {best_score:.3f}')
        ax.legend()
        
        plt.tight_layout()
        
        # Save
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved ablation plot: {save_path}")
        return str(save_path)
    
    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        class_names: List[str],
        save_name: str = "confusion_matrix.png",
        title: str = "Confusion Matrix",
        normalize: bool = False
    ) -> Optional[str]:
        """
        Plot confusion matrix heatmap.
        
        Args:
            cm: Confusion matrix (2D array)
            class_names: List of class names
            save_name: Output filename
            title: Plot title
            normalize: Whether to normalize by row
        
        Returns:
            Path to saved plot
        """
        if not PLOTTING_AVAILABLE:
            return None
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Create heatmap
        sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd',
                   cmap='Blues', square=True, cbar_kws={'label': 'Count'},
                   xticklabels=class_names, yticklabels=class_names,
                   ax=ax, linewidths=0.5, linecolor='gray')
        
        # Customize
        ax.set_xlabel('Predicted Label', fontweight='bold')
        ax.set_ylabel('True Label', fontweight='bold')
        ax.set_title(title, fontweight='bold')
        
        plt.tight_layout()
        
        # Save
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved confusion matrix: {save_path}")
        return str(save_path)
    
    def plot_roc_curve(
        self,
        fpr: np.ndarray,
        tpr: np.ndarray,
        auc: float,
        save_name: str = "roc_curve.png",
        title: str = "ROC Curve"
    ) -> Optional[str]:
        """
        Plot ROC curve.
        
        Args:
            fpr: False positive rates
            tpr: True positive rates
            auc: Area under curve
            save_name: Output filename
            title: Plot title
        
        Returns:
            Path to saved plot
        """
        if not PLOTTING_AVAILABLE:
            return None
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot ROC curve
        ax.plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {auc:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        
        # Customize
        ax.set_xlabel('False Positive Rate', fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontweight='bold')
        ax.set_title(title, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        plt.tight_layout()
        
        # Save
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved ROC curve: {save_path}")
        return str(save_path)
    
    def plot_learning_curve(
        self,
        train_scores: List[float],
        val_scores: List[float],
        metric: str = "Loss",
        save_name: str = "learning_curve.png",
        title: Optional[str] = None
    ) -> Optional[str]:
        """
        Plot learning curve (training vs validation).
        
        Args:
            train_scores: Training scores per epoch
            val_scores: Validation scores per epoch
            metric: Metric name
            save_name: Output filename
            title: Plot title
        
        Returns:
            Path to saved plot
        """
        if not PLOTTING_AVAILABLE:
            return None
        
        epochs = np.arange(1, len(train_scores) + 1)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot curves
        ax.plot(epochs, train_scores, marker='o', label='Training', linewidth=2)
        ax.plot(epochs, val_scores, marker='s', label='Validation', linewidth=2)
        
        # Customize
        ax.set_xlabel('Epoch', fontweight='bold')
        ax.set_ylabel(metric, fontweight='bold')
        ax.set_title(title or f'Learning Curve - {metric}', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved learning curve: {save_path}")
        return str(save_path)
    
    def plot_feature_importance(
        self,
        features: List[str],
        importances: List[float],
        save_name: str = "feature_importance.png",
        title: str = "Feature Importance",
        top_n: Optional[int] = None
    ) -> Optional[str]:
        """
        Plot feature importance bar chart.
        
        Args:
            features: Feature names
            importances: Importance scores
            save_name: Output filename
            title: Plot title
            top_n: Show only top N features
        
        Returns:
            Path to saved plot
        """
        if not PLOTTING_AVAILABLE:
            return None
        
        # Sort by importance
        sorted_idx = np.argsort(importances)[::-1]
        
        if top_n:
            sorted_idx = sorted_idx[:top_n]
        
        sorted_features = [features[i] for i in sorted_idx]
        sorted_importances = [importances[i] for i in sorted_idx]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, max(6, len(sorted_features) * 0.3)))
        
        # Horizontal bar chart
        y_pos = np.arange(len(sorted_features))
        ax.barh(y_pos, sorted_importances, alpha=0.8, edgecolor='black')
        
        # Customize
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_features)
        ax.invert_yaxis()
        ax.set_xlabel('Importance', fontweight='bold')
        ax.set_title(title, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        # Save
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved feature importance plot: {save_path}")
        return str(save_path)
