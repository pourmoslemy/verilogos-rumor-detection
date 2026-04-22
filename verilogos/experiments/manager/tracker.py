"""
Experiment Tracker - Local W&B-style tracking system

Logs experiments to timestamped directories with metrics, configs, and plots.
No external dependencies required.

Author: Alireza Pourmoslemi
Email: apmath99@gmail.com
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import csv


class ExperimentTracker:
    """
    Local experiment tracking system similar to Weights & Biases.
    
    Features:
    - Timestamped experiment folders
    - Config logging
    - JSONL metrics logging
    - CSV export
    - Automatic plot storage
    - Reproducible experiment metadata
    
    Example:
        >>> tracker = ExperimentTracker(base_dir="results")
        >>> tracker.start_experiment("topology_baseline")
        >>> tracker.log_config({"model": "RF", "features": "topology"})
        >>> tracker.log_metrics({"accuracy": 0.95, "f1": 0.94})
        >>> tracker.finish_experiment()
    """
    
    def __init__(self, base_dir: str = "results"):
        """
        Initialize experiment tracker.
        
        Args:
            base_dir: Base directory for all experiments
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        self.current_experiment = None
        self.experiment_dir = None
        self.metrics_file = None
        self.config = {}
        self.metrics_history = []
        self.start_time = None
        
    def start_experiment(self, name: str, tags: Optional[List[str]] = None) -> str:
        """
        Start a new experiment.
        
        Args:
            name: Experiment name
            tags: Optional tags for categorization
            
        Returns:
            Experiment ID (timestamp-based)
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_id = f"{name}_{timestamp}"
        
        self.current_experiment = experiment_id
        self.experiment_dir = self.base_dir / experiment_id
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.experiment_dir / "plots").mkdir(exist_ok=True)
        (self.experiment_dir / "checkpoints").mkdir(exist_ok=True)
        
        # Initialize metrics file
        self.metrics_file = self.experiment_dir / "metrics.jsonl"
        
        # Log metadata
        self.start_time = time.time()
        metadata = {
            "experiment_id": experiment_id,
            "name": name,
            "tags": tags or [],
            "start_time": datetime.now().isoformat(),
            "status": "running"
        }
        
        with open(self.experiment_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Started experiment: {experiment_id}")
        print(f"Results directory: {self.experiment_dir}")
        
        return experiment_id
    
    def log_config(self, config: Dict[str, Any]):
        """
        Log experiment configuration.
        
        Args:
            config: Configuration dictionary
        """
        if not self.current_experiment:
            raise ValueError("No active experiment. Call start_experiment() first.")
        
        self.config = config
        
        config_file = self.experiment_dir / "config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Logged config: {len(config)} parameters")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log metrics for current step.
        
        Args:
            metrics: Dictionary of metric name -> value
            step: Optional step number (epoch, iteration, etc.)
        """
        if not self.current_experiment:
            raise ValueError("No active experiment. Call start_experiment() first.")
        
        # Add timestamp and step
        log_entry = {
            "timestamp": time.time(),
            "step": step if step is not None else len(self.metrics_history),
            **metrics
        }
        
        # Append to JSONL file
        with open(self.metrics_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
        
        self.metrics_history.append(log_entry)
    
    def log_artifact(self, filepath: str, artifact_type: str = "file"):
        """
        Log an artifact (file, plot, checkpoint).
        
        Args:
            filepath: Path to artifact file
            artifact_type: Type of artifact (file, plot, checkpoint)
        """
        if not self.current_experiment:
            raise ValueError("No active experiment. Call start_experiment() first.")
        
        source = Path(filepath)
        if not source.exists():
            print(f"Warning: Artifact not found: {filepath}")
            return
        
        # Determine destination
        if artifact_type == "plot":
            dest_dir = self.experiment_dir / "plots"
        elif artifact_type == "checkpoint":
            dest_dir = self.experiment_dir / "checkpoints"
        else:
            dest_dir = self.experiment_dir
        
        dest = dest_dir / source.name
        
        # Copy file
        import shutil
        shutil.copy2(source, dest)
        
        print(f"Logged {artifact_type}: {source.name}")
    
    def finish_experiment(self, status: str = "completed"):
        """
        Finish current experiment and export results.
        
        Args:
            status: Final status (completed, failed, stopped)
        """
        if not self.current_experiment:
            print("Warning: No active experiment to finish.")
            return
        
        # Calculate duration
        duration = time.time() - self.start_time if self.start_time else 0
        
        # Update metadata
        metadata_file = self.experiment_dir / "metadata.json"
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        metadata.update({
            "status": status,
            "end_time": datetime.now().isoformat(),
            "duration_seconds": duration
        })
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Export summary
        self._export_summary()
        
        # Export metrics to CSV
        self._export_metrics_csv()
        
        print(f"Finished experiment: {self.current_experiment}")
        print(f"Status: {status}")
        print(f"Duration: {duration:.2f}s")
        
        # Reset state
        self.current_experiment = None
        self.experiment_dir = None
        self.metrics_file = None
        self.config = {}
        self.metrics_history = []
        self.start_time = None
    
    def _export_summary(self):
        """Export experiment summary with final metrics."""
        if not self.metrics_history:
            return
        
        # Get final metrics
        final_metrics = self.metrics_history[-1].copy()
        final_metrics.pop('timestamp', None)
        final_metrics.pop('step', None)
        
        summary = {
            "experiment_id": self.current_experiment,
            "config": self.config,
            "final_metrics": final_metrics,
            "total_steps": len(self.metrics_history)
        }
        
        summary_file = self.experiment_dir / "summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
    
    def _export_metrics_csv(self):
        """Export metrics history to CSV."""
        if not self.metrics_history:
            return
        
        csv_file = self.experiment_dir / "metrics.csv"
        
        # Get all metric keys
        all_keys = set()
        for entry in self.metrics_history:
            all_keys.update(entry.keys())
        
        fieldnames = sorted(all_keys)
        
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.metrics_history)
    
    def get_experiment_dir(self) -> Optional[Path]:
        """Get current experiment directory."""
        return self.experiment_dir
    
    def list_experiments(self) -> List[Dict[str, Any]]:
        """
        List all experiments in base directory.
        
        Returns:
            List of experiment metadata dictionaries
        """
        experiments = []
        
        for exp_dir in self.base_dir.iterdir():
            if not exp_dir.is_dir():
                continue
            
            metadata_file = exp_dir / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    experiments.append(metadata)
        
        # Sort by start time
        experiments.sort(key=lambda x: x.get('start_time', ''), reverse=True)
        
        return experiments
    
    def load_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """
        Load experiment data.
        
        Args:
            experiment_id: Experiment ID to load
            
        Returns:
            Dictionary with metadata, config, and metrics
        """
        exp_dir = self.base_dir / experiment_id
        
        if not exp_dir.exists():
            raise ValueError(f"Experiment not found: {experiment_id}")
        
        # Load metadata
        with open(exp_dir / "metadata.json", 'r') as f:
            metadata = json.load(f)
        
        # Load config
        config_file = exp_dir / "config.json"
        config = {}
        if config_file.exists():
            with open(config_file, 'r') as f:
                config = json.load(f)
        
        # Load metrics
        metrics_file = exp_dir / "metrics.jsonl"
        metrics = []
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                for line in f:
                    metrics.append(json.loads(line))
        
        return {
            "metadata": metadata,
            "config": config,
            "metrics": metrics
        }
