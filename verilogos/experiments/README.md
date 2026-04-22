# VeriLogos NeurIPS-Grade Experiment Manager

**Author:** Alireza Pourmoslemi  
**Email:** apmath99@gmail.com  
**Version:** 1.0.0

---

## Overview

Complete research-grade experiment management system for VeriLogos. Designed for reproducible, scalable experiments with automatic tracking, parallel execution, and publication-quality visualizations.

### Features

- **Experiment Tracking**: Local W&B-style tracking with timestamped directories
- **Parallel Execution**: Multi-process experiment runner with fault tolerance
- **Ablation Studies**: Automatic generation of all parameter combinations
- **Publication Plots**: Paper-ready visualizations (matplotlib + seaborn)
- **Multiple Datasets**: FakeNewsNet, LIAR, PHEME support
- **Multiple Models**: Classical baselines + VeriLogos topology + Hybrid
- **Comprehensive Metrics**: Accuracy, Precision, Recall, F1, ROC-AUC
- **YAML Configuration**: Easy experiment definition

---

## Quick Start

### Run All Experiments

```bash
cd /mnt/d/VeriLogos
python run_experiments.py
```

This will:
1. Load configuration from `verilogos/experiments/config/experiments.yaml`
2. Generate all dataset × model × ablation combinations
3. Run experiments in parallel (4 workers by default)
4. Save results to `results/` directory
5. Generate plots automatically

### Run Specific Dataset

```bash
python run_experiments.py --dataset FakeNewsNet
```

### Run Specific Model

```bash
python run_experiments.py --model RandomForest
```

### Sequential Mode (Debugging)

```bash
python run_experiments.py --sequential
```

---

## Project Structure

```
verilogos/experiments/
├── manager/
│   ├── tracker.py              # Experiment tracking (200 lines)
│   ├── parallel_executor.py    # Parallel execution (150 lines)
│   ├── ablation.py             # Ablation generator (100 lines)
│   ├── plotting.py             # Publication plots (300 lines)
│   └── experiment_manager.py   # Main orchestrator (400 lines)
│
├── datasets/
│   ├── base.py                 # BaseDataset interface
│   ├── fakenewsnet.py          # FakeNewsNet loader
│   ├── liar.py                 # LIAR dataset
│   └── pheme.py                # PHEME dataset
│
├── models/
│   ├── base.py                 # BaseModel interface
│   ├── classical.py            # LR, RF, SVM, XGBoost
│   ├── verilogos_topology.py  # Pure topology model
│   └── hybrid.py               # Topology + Text
│
├── features/
│   ├── topology_features.py    # Betti, persistence
│   └── text_features.py        # TF-IDF, embeddings
│
├── evaluation/
│   ├── metrics.py              # Comprehensive metrics
│   └── visualization.py        # Plotting utilities
│
├── config/
│   └── experiments.yaml        # Experiment configuration
│
└── runner.py                   # Single experiment executor
```

**Total:** ~3,500 lines of production code across 25+ files

---

## Configuration

Edit `verilogos/experiments/config/experiments.yaml`:

```yaml
# Datasets to evaluate
datasets:
  - FakeNewsNet
  - LIAR
  - PHEME

# Models to test
models:
  - LogisticRegression
  - RandomForest
  - SVM
  - XGBoost
  - VeriLogosTopology
  - Hybrid

# Ablation study
ablations:
  topology: [true, false]
  text: [true, false]

# Parallel settings
parallel:
  max_workers: 4
  batch_size: 10
```

---

## Usage Examples

### Basic Usage

```python
from verilogos.experiments import ExperimentManager

# Initialize
manager = ExperimentManager("config/experiments.yaml")

# Run all experiments
results = manager.run_all_experiments()

# Export results
manager.export_results_csv("results.csv")
```

### Custom Ablation Study

```python
from verilogos.experiments.manager import AblationGenerator

generator = AblationGenerator()

# Define ablation parameters
config = {
    "topology": [True, False],
    "bert": [True, False],
    "temporal": [True, False]
}

# Generate all combinations (2^3 = 8)
ablations = generator.generate(config)
```

### Parallel Execution

```python
from verilogos.experiments.manager import ParallelExecutor

executor = ParallelExecutor(max_workers=4)

# Define tasks
tasks = [
    {"model": "RF", "dataset": "FakeNewsNet"},
    {"model": "SVM", "dataset": "LIAR"},
    # ...
]

# Run in parallel
results = executor.run(train_model, tasks)
```

### Experiment Tracking

```python
from verilogos.experiments.manager import ExperimentTracker

tracker = ExperimentTracker(base_dir="results")

# Start experiment
tracker.start_experiment("topology_baseline")

# Log configuration
tracker.log_config({"model": "RF", "features": "topology"})

# Log metrics
tracker.log_metrics({"accuracy": 0.95, "f1": 0.94})

# Finish
tracker.finish_experiment()
```

---

## Output

### Results Directory Structure

```
results/
├── topology_baseline_20250420_120000/
│   ├── metadata.json           # Experiment metadata
│   ├── config.json             # Configuration
│   ├── metrics.jsonl           # Metrics log (one line per step)
│   ├── metrics.csv             # Metrics in CSV format
│   ├── summary.json            # Final summary
│   └── plots/
│       ├── model_comparison_f1.png
│       ├── model_comparison_accuracy.png
│       ├── ablation_study.png
│       └── confusion_matrix.png
│
├── all_results.json            # All experiment results
├── summary.json                # Overall summary
└── results.csv                 # Exportable CSV
```

### Metrics Computed

- **Accuracy**: Overall correctness
- **Precision**: True positive rate
- **Recall**: Sensitivity
- **F1 Score**: Harmonic mean
- **ROC-AUC**: Area under ROC curve
- **Confusion Matrix**: TP, FP, TN, FN

---

## Models

### Classical Baselines

- **Logistic Regression**: Linear classifier
- **Random Forest**: Ensemble of decision trees
- **SVM**: Support vector machine with RBF kernel
- **XGBoost**: Gradient boosting

### VeriLogos Models

- **VeriLogosTopology**: Pure topology features (Betti numbers, persistence)
- **Hybrid**: Combines topology + text features

### Feature Types

**Topology Features:**
- Betti numbers (B0, B1, B2)
- Persistence statistics (max, mean, std, entropy)
- Euler characteristic
- Fragmentation index

**Text Features:**
- TF-IDF vectors (up to 1000 features)
- N-grams (1-2)
- Basic statistics (word count, sentence count, etc.)

---

## Datasets

### FakeNewsNet

- **Source**: Politifact + GossipCop
- **Format**: CSV or JSON
- **Labels**: Real (0), Fake (1)
- **Synthetic**: Auto-generated if file not found

### LIAR

- **Source**: LIAR dataset (6-way labels collapsed to binary)
- **Format**: TSV
- **Labels**: True/Mostly-true/Half-true → Real, False/Barely-true/Pants-fire → Fake

### PHEME

- **Source**: Twitter rumors
- **Format**: JSON (directory structure or single file)
- **Labels**: Rumor (1), Non-rumor (0)

---

## Advanced Features

### Batch Execution

```python
from verilogos.experiments.manager import BatchExecutor

executor = BatchExecutor(max_workers=4, batch_size=10)
results = executor.run(train_model, tasks)
```

### Incremental Ablation

```python
generator = AblationGenerator()

# Add one feature at a time
configs = generator.generate_incremental(
    features=['topology', 'text', 'temporal'],
    start_with_all=False
)
```

### Custom Plotting

```python
from verilogos.experiments.manager import ExperimentPlotter

plotter = ExperimentPlotter(output_dir="plots")

# Model comparison
plotter.plot_model_comparison(results, metric='f1')

# Ablation study
plotter.plot_ablation(ablation_results)

# Feature importance
plotter.plot_feature_importance(features, importances)
```

---

## Performance

### Benchmarks

- **Single Experiment**: ~10-30 seconds (depending on dataset size)
- **Full Suite** (3 datasets × 6 models × 4 ablations = 72 experiments):
  - Sequential: ~30-60 minutes
  - Parallel (4 workers): ~10-20 minutes

### Scalability

- Supports up to CPU count workers
- Fault-tolerant: Failed experiments don't stop the suite
- Memory-efficient: Batch processing for large datasets

---

## Troubleshooting

### Import Errors

```bash
export PYTHONPATH=/mnt/d/VeriLogos:$PYTHONPATH
```

### Missing Dependencies

```bash
pip install scikit-learn numpy matplotlib seaborn pyyaml
pip install xgboost  # Optional
```

### Dataset Not Found

The system will automatically generate synthetic data if dataset files are not found. To use real data, update `experiments.yaml`:

```yaml
data_paths:
  fakenewsnet: /path/to/fakenewsnet.csv
  liar: /path/to/liar_train.tsv
  pheme: /path/to/pheme_dir
```

### Network Issues

If downloading datasets fails, enable SOCKS proxy:

```bash
# Configure proxy at 127.0.0.1:1080
export http_proxy=socks5://127.0.0.1:1080
export https_proxy=socks5://127.0.0.1:1080
```

---

## Citation

If you use this experiment framework, please cite:

```bibtex
@software{verilogos_experiments,
  author = {Pourmoslemi, Alireza},
  title = {VeriLogos NeurIPS-Grade Experiment Manager},
  year = {2025},
  email = {apmath99@gmail.com}
}
```

---

## License

MIT License

Copyright (c) 2025 Alireza Pourmoslemi

---

## Contact

**Author:** Alireza Pourmoslemi  
**Email:** apmath99@gmail.com  
**Project:** VeriLogos - Topological Market Regime Detection

---

**Status:** ✅ Production Ready | 🚀 NeurIPS-Grade | 📊 Publication Quality
