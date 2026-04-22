"""
Publication-Quality Visualization for Academic Paper
Generates ROC curves, PR curves, confusion matrices, and comparison plots
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report, f1_score
)
from typing import Dict, List
from pathlib import Path
import json


# Set publication-quality style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9


class ResultsVisualizer:
    """
    Generates publication-quality plots for academic paper.
    """
    
    def __init__(self, output_dir: str = './results'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_roc_curves(
        self,
        results: Dict[str, Dict],
        save_name: str = 'roc_curves.png'
    ):
        """
        Plot ROC curves for all models on the same figure.
        
        Args:
            results: Dict mapping model_name to evaluation results
                     Each result must have 'labels' and 'probabilities'
            save_name: Output filename
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        for i, (model_name, result) in enumerate(results.items()):
            labels = np.array(result['labels'])
            # Get probability of positive class (class 1)
            probs = np.array(result['probabilities'])[:, 1]
            
            # Compute ROC curve
            fpr, tpr, _ = roc_curve(labels, probs)
            roc_auc = auc(fpr, tpr)
            
            # Plot
            ax.plot(
                fpr, tpr,
                color=colors[i % len(colors)],
                lw=2,
                label=f'{model_name} (AUC = {roc_auc:.3f})'
            )
        
        # Diagonal reference line
        ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random Classifier')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves - Fake News Detection')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
        print(f"Saved ROC curves to {save_path}")
    
    def plot_precision_recall_curves(
        self,
        results: Dict[str, Dict],
        save_name: str = 'pr_curves.png'
    ):
        """
        Plot Precision-Recall curves for all models.
        
        Args:
            results: Dict mapping model_name to evaluation results
            save_name: Output filename
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        for i, (model_name, result) in enumerate(results.items()):
            labels = np.array(result['labels'])
            probs = np.array(result['probabilities'])[:, 1]
            
            # Compute PR curve
            precision, recall, _ = precision_recall_curve(labels, probs)
            avg_precision = average_precision_score(labels, probs)
            
            # Plot
            ax.plot(
                recall, precision,
                color=colors[i % len(colors)],
                lw=2,
                label=f'{model_name} (AP = {avg_precision:.3f})'
            )
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curves - Fake News Detection')
        ax.legend(loc='lower left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
        print(f"Saved PR curves to {save_path}")
    
    def plot_confusion_matrix(
        self,
        labels: List[int],
        predictions: List[int],
        model_name: str,
        save_name: str = None
    ):
        """
        Plot confusion matrix heatmap.
        
        Args:
            labels: True labels
            predictions: Predicted labels
            model_name: Name of the model
            save_name: Output filename (auto-generated if None)
        """
        if save_name is None:
            save_name = f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png'
        
        cm = confusion_matrix(labels, predictions)
        
        fig, ax = plt.subplots(figsize=(6, 5))
        
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Fake', 'Real'],
            yticklabels=['Fake', 'Real'],
            ax=ax,
            cbar_kws={'label': 'Count'}
        )
        
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title(f'Confusion Matrix - {model_name}')
        
        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
        print(f"Saved confusion matrix to {save_path}")
    
    def plot_metrics_comparison(
        self,
        results: Dict[str, Dict],
        save_name: str = 'metrics_comparison.png'
    ):
        """
        Plot bar chart comparing F1, Precision, Recall, Accuracy across models.
        
        Args:
            results: Dict mapping model_name to evaluation results
            save_name: Output filename
        """
        from sklearn.metrics import precision_score, recall_score
        
        models = list(results.keys())
        metrics = {
            'Accuracy': [],
            'Precision': [],
            'Recall': [],
            'F1-Score': []
        }
        
        for model_name in models:
            result = results[model_name]
            labels = result['labels']
            predictions = result['predictions']
            
            metrics['Accuracy'].append(result['accuracy'])
            metrics['Precision'].append(precision_score(labels, predictions, average='weighted'))
            metrics['Recall'].append(recall_score(labels, predictions, average='weighted'))
            metrics['F1-Score'].append(f1_score(labels, predictions, average='weighted'))
        
        # Create grouped bar chart
        x = np.arange(len(models))
        width = 0.2
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for i, (metric_name, values) in enumerate(metrics.items()):
            offset = width * (i - 1.5)
            ax.bar(x + offset, values, width, label=metric_name, color=colors[i])
        
        ax.set_xlabel('Model')
        ax.set_ylabel('Score')
        ax.set_title('Performance Metrics Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.legend()
        ax.set_ylim([0, 1.1])
        ax.grid(True, axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, (metric_name, values) in enumerate(metrics.items()):
            offset = width * (i - 1.5)
            for j, v in enumerate(values):
                ax.text(j + offset, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
        print(f"Saved metrics comparison to {save_path}")
    
    def plot_training_history(
        self,
        history: Dict,
        model_name: str,
        save_name: str = None
    ):
        """
        Plot training and validation loss/accuracy curves.
        
        Args:
            history: Training history dict with 'train_loss', 'val_loss', etc.
            model_name: Name of the model
            save_name: Output filename (auto-generated if None)
        """
        if save_name is None:
            save_name = f'training_history_{model_name.lower().replace(" ", "_")}.png'
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        # Loss plot
        ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title(f'Training History - {model_name}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy plot
        ax2.plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
        ax2.plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title(f'Accuracy History - {model_name}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
        print(f"Saved training history to {save_path}")
    
    def generate_classification_report(
        self,
        results: Dict[str, Dict],
        save_name: str = 'classification_report.txt'
    ):
        """
        Generate detailed classification report for all models.
        
        Args:
            results: Dict mapping model_name to evaluation results
            save_name: Output filename
        """
        report_lines = []
        report_lines.append("="*80)
        report_lines.append("CLASSIFICATION REPORT - FAKE NEWS DETECTION")
        report_lines.append("="*80)
        report_lines.append("")
        
        for model_name, result in results.items():
            labels = result['labels']
            predictions = result['predictions']
            
            report_lines.append(f"\n{model_name}")
            report_lines.append("-" * 80)
            
            # Classification report
            report = classification_report(
                labels, predictions,
                target_names=['Fake', 'Real'],
                digits=4
            )
            report_lines.append(report)
            
            # Confusion matrix
            cm = confusion_matrix(labels, predictions)
            report_lines.append("\nConfusion Matrix:")
            report_lines.append(f"              Predicted")
            report_lines.append(f"              Fake  Real")
            report_lines.append(f"Actual Fake   {cm[0][0]:4d}  {cm[0][1]:4d}")
            report_lines.append(f"       Real   {cm[1][0]:4d}  {cm[1][1]:4d}")
            report_lines.append("")
        
        # Save report
        save_path = self.output_dir / save_name
        with open(save_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"Saved classification report to {save_path}")
        
        # Also print to console
        print('\n'.join(report_lines))
    
    def save_results_json(
        self,
        results: Dict[str, Dict],
        save_name: str = 'results.json'
    ):
        """
        Save numerical results to JSON for further analysis.
        
        Args:
            results: Dict mapping model_name to evaluation results
            save_name: Output filename
        """
        from sklearn.metrics import precision_score, recall_score
        
        output = {}
        
        for model_name, result in results.items():
            labels = result['labels']
            predictions = result['predictions']
            probs = np.array(result['probabilities'])[:, 1]
            
            # Compute all metrics
            fpr, tpr, _ = roc_curve(labels, probs)
            roc_auc = auc(fpr, tpr)
            avg_precision = average_precision_score(labels, probs)
            
            output[model_name] = {
                'accuracy': float(result['accuracy']),
                'precision': float(precision_score(labels, predictions, average='weighted')),
                'recall': float(recall_score(labels, predictions, average='weighted')),
                'f1_score': float(f1_score(labels, predictions, average='weighted')),
                'roc_auc': float(roc_auc),
                'average_precision': float(avg_precision)
            }
        
        save_path = self.output_dir / save_name
        with open(save_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"Saved results JSON to {save_path}")


def generate_all_visualizations(
    results: Dict[str, Dict],
    training_histories: Dict[str, Dict],
    output_dir: str = './results'
):
    """
    Generate all publication-quality visualizations.
    
    Args:
        results: Dict mapping model_name to evaluation results
        training_histories: Dict mapping model_name to training history
        output_dir: Output directory for plots
    """
    visualizer = ResultsVisualizer(output_dir)
    
    print("\nGenerating publication-quality visualizations...")
    
    # ROC curves
    visualizer.plot_roc_curves(results)
    
    # Precision-Recall curves
    visualizer.plot_precision_recall_curves(results)
    
    # Confusion matrices for each model
    for model_name, result in results.items():
        visualizer.plot_confusion_matrix(
            result['labels'],
            result['predictions'],
            model_name
        )
    
    # Metrics comparison
    visualizer.plot_metrics_comparison(results)
    
    # Training histories
    for model_name, history in training_histories.items():
        if history:  # Skip if no history available
            visualizer.plot_training_history(history, model_name)
    
    # Classification report
    visualizer.generate_classification_report(results)
    
    # Save JSON results
    visualizer.save_results_json(results)
    
    print(f"\nAll visualizations saved to {output_dir}/")
