"""
Experiment Manager - Main orchestration layer

Coordinates datasets, models, ablations, and parallel execution.

Author: Alireza Pourmoslemi
Email: apmath99@gmail.com
"""

import yaml
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import time

from verilogos.experiments.manager.tracker import ExperimentTracker
from verilogos.experiments.manager.parallel_executor import ParallelExecutor
from verilogos.experiments.manager.ablation import AblationGenerator
from verilogos.experiments.manager.plotting import ExperimentPlotter


class ExperimentManager:
    """
    Main experiment orchestration system.
    
    Responsibilities:
    - Load configuration
    - Generate ablation combinations
    - Run dataset × model × ablation experiments
    - Use ParallelExecutor for parallel training
    - Collect metrics
    - Log results through ExperimentTracker
    - Generate plots automatically
    
    Example:
        >>> manager = ExperimentManager("config/experiments.yaml")
        >>> manager.run_all_experiments()
    """
    
    def __init__(
        self,
        config_path: str,
        results_dir: str = "results",
        max_workers: Optional[int] = None
    ):
        """
        Initialize experiment manager.
        
        Args:
            config_path: Path to YAML configuration file
            results_dir: Directory for results
            max_workers: Maximum parallel workers
        """
        self.config_path = Path(config_path)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize components
        self.tracker = ExperimentTracker(base_dir=str(self.results_dir))
        self.executor = ParallelExecutor(max_workers=max_workers)
        self.ablation_gen = AblationGenerator()
        self.plotter = ExperimentPlotter(output_dir=str(self.results_dir / "plots"))
        
        # Storage for results
        self.all_results = []
        
        print(f"ExperimentManager initialized")
        print(f"Config: {self.config_path}")
        print(f"Results: {self.results_dir}")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load experiment configuration from YAML."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        print(f"Loaded configuration from {self.config_path}")
        return config
    
    def generate_experiment_tasks(self) -> List[Dict[str, Any]]:
        """
        Generate all experiment tasks from configuration.
        
        Returns:
            List of experiment task configurations
        """
        tasks = []
        
        # Get datasets and models from config
        datasets = self.config.get('datasets', [])
        models = self.config.get('models', [])
        
        # Generate ablations if specified
        ablation_config = self.config.get('ablations', {})
        if ablation_config:
            ablations = self.ablation_gen.generate(ablation_config)
        else:
            ablations = [{}]  # Single baseline
        
        # Create all combinations
        task_id = 0
        for dataset in datasets:
            for model in models:
                for ablation in ablations:
                    task = {
                        'task_id': task_id,
                        'dataset': dataset,
                        'model': model,
                        'ablation': ablation,
                        'config': self.config
                    }
                    tasks.append(task)
                    task_id += 1
        
        print(f"\nGenerated {len(tasks)} experiment tasks:")
        print(f"  Datasets: {len(datasets)}")
        print(f"  Models: {len(models)}")
        print(f"  Ablations: {len(ablations)}")
        
        return tasks
    
    def run_all_experiments(self, parallel: bool = True) -> List[Dict[str, Any]]:
        """
        Run all experiments defined in configuration.
        
        Args:
            parallel: Whether to run in parallel
        
        Returns:
            List of all experiment results
        """
        print("\n" + "="*80)
        print("STARTING EXPERIMENT SUITE")
        print("="*80)
        
        start_time = time.time()
        
        # Generate tasks
        tasks = self.generate_experiment_tasks()
        
        if not tasks:
            print("No tasks to execute")
            return []
        
        # Import the experiment runner function
        from verilogos.experiments.runner import run_single_experiment
        
        # Execute tasks
        if parallel:
            results = self.executor.run(run_single_experiment, tasks)
        else:
            results = self.executor.run_sequential(run_single_experiment, tasks)
        
        # Store results
        self.all_results = results
        
        # Generate summary
        self._generate_summary(results)
        
        # Generate plots
        self._generate_plots(results)
        
        # Save all results
        self._save_results(results)
        
        duration = time.time() - start_time
        
        print("\n" + "="*80)
        print("EXPERIMENT SUITE COMPLETE")
        print("="*80)
        print(f"Total time: {duration:.2f}s")
        print(f"Results saved to: {self.results_dir}")
        
        return results
    
    def _generate_summary(self, results: List[Dict[str, Any]]):
        """Generate experiment summary statistics."""
        print("\n" + "="*80)
        print("EXPERIMENT SUMMARY")
        print("="*80)
        
        successful = [r for r in results if r.get('status') == 'success']
        failed = [r for r in results if r.get('status') == 'failed']
        
        print(f"Total experiments: {len(results)}")
        print(f"Successful: {len(successful)}")
        print(f"Failed: {len(failed)}")
        
        if successful:
            # Find best result
            best_result = max(successful, key=lambda x: x.get('f1', 0))
            print(f"\nBest result:")
            print(f"  Model: {best_result.get('model', 'unknown')}")
            print(f"  Dataset: {best_result.get('dataset', 'unknown')}")
            print(f"  F1 Score: {best_result.get('f1', 0):.4f}")
            print(f"  Accuracy: {best_result.get('accuracy', 0):.4f}")
        
        if failed:
            print(f"\nFailed experiments:")
            for r in failed[:5]:  # Show first 5
                print(f"  Task {r.get('task_idx')}: {r.get('error', 'unknown error')}")
    
    def _generate_plots(self, results: List[Dict[str, Any]]):
        """Generate all plots from results."""
        print("\n" + "="*80)
        print("GENERATING PLOTS")
        print("="*80)
        
        successful = [r for r in results if r.get('status') == 'success']
        
        if not successful:
            print("No successful results to plot")
            return
        
        # Model comparison plot
        try:
            self.plotter.plot_model_comparison(
                successful,
                metric='f1',
                save_name='model_comparison_f1.png'
            )
            self.plotter.plot_model_comparison(
                successful,
                metric='accuracy',
                save_name='model_comparison_accuracy.png'
            )
        except Exception as e:
            print(f"Error generating model comparison: {e}")
        
        # Ablation plot if ablations were run
        ablation_config = self.config.get('ablations', {})
        if ablation_config:
            try:
                self.plotter.plot_ablation(
                    successful,
                    metric='f1',
                    save_name='ablation_study.png'
                )
            except Exception as e:
                print(f"Error generating ablation plot: {e}")
    
    def _save_results(self, results: List[Dict[str, Any]]):
        """Save all results to files."""
        # Save as JSON
        results_file = self.results_dir / "all_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nSaved results to {results_file}")
        
        # Save summary statistics
        successful = [r for r in results if r.get('status') == 'success']
        
        if successful:
            summary = {
                'total_experiments': len(results),
                'successful': len(successful),
                'failed': len(results) - len(successful),
                'best_f1': max(r.get('f1', 0) for r in successful),
                'best_accuracy': max(r.get('accuracy', 0) for r in successful),
                'avg_f1': sum(r.get('f1', 0) for r in successful) / len(successful),
                'avg_accuracy': sum(r.get('accuracy', 0) for r in successful) / len(successful)
            }
            
            summary_file = self.results_dir / "summary.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            print(f"Saved summary to {summary_file}")
    
    def run_single_dataset(self, dataset_name: str, parallel: bool = True) -> List[Dict[str, Any]]:
        """
        Run experiments for a single dataset.
        
        Args:
            dataset_name: Name of dataset
            parallel: Whether to run in parallel
        
        Returns:
            List of results for this dataset
        """
        # Filter tasks for this dataset
        all_tasks = self.generate_experiment_tasks()
        dataset_tasks = [t for t in all_tasks if t['dataset'] == dataset_name]
        
        if not dataset_tasks:
            print(f"No tasks found for dataset: {dataset_name}")
            return []
        
        print(f"\nRunning {len(dataset_tasks)} experiments for dataset: {dataset_name}")
        
        from verilogos.experiments.runner import run_single_experiment
        
        if parallel:
            results = self.executor.run(run_single_experiment, dataset_tasks)
        else:
            results = self.executor.run_sequential(run_single_experiment, dataset_tasks)
        
        return results
    
    def run_single_model(self, model_name: str, parallel: bool = True) -> List[Dict[str, Any]]:
        """
        Run experiments for a single model.
        
        Args:
            model_name: Name of model
            parallel: Whether to run in parallel
        
        Returns:
            List of results for this model
        """
        all_tasks = self.generate_experiment_tasks()
        model_tasks = [t for t in all_tasks if t['model'] == model_name]
        
        if not model_tasks:
            print(f"No tasks found for model: {model_name}")
            return []
        
        print(f"\nRunning {len(model_tasks)} experiments for model: {model_name}")
        
        from verilogos.experiments.runner import run_single_experiment
        
        if parallel:
            results = self.executor.run(run_single_experiment, model_tasks)
        else:
            results = self.executor.run_sequential(run_single_experiment, model_tasks)
        
        return results
    
    def get_results(self) -> List[Dict[str, Any]]:
        """Get all experiment results."""
        return self.all_results
    
    def export_results_csv(self, filename: str = "results.csv"):
        """
        Export results to CSV.
        
        Args:
            filename: Output CSV filename
        """
        import csv
        
        if not self.all_results:
            print("No results to export")
            return
        
        # Get all keys
        all_keys = set()
        for result in self.all_results:
            all_keys.update(result.keys())
        
        fieldnames = sorted(all_keys)
        
        csv_path = self.results_dir / filename
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.all_results)
        
        print(f"Exported results to {csv_path}")
