# Topological Lie Detector

A production-grade misinformation detection system using topological data analysis and persistent homology.

## Overview

This system detects fake news by analyzing the topological structure of narrative graphs. It transforms text into simplicial complexes, computes persistent homology features, and uses machine learning to classify content as real or fake.

**Author:** Alireza Pourmoslemi (apmath99@gmail.com)

## Architecture

```
Text Input
    ↓
Text Preprocessing
    ↓
Graph Construction (entities, claims, relations)
    ↓
Simplicial Complex Builder
    ↓
Persistent Homology Computation
    ↓
Topological Feature Extraction
    ↓
ML Classifier (Random Forest / XGBoost)
    ↓
Prediction + Explanation
```

## Features

- **Truth Geometry Analysis**: Computes Betti numbers, Euler characteristic, persistence barcodes
- **Shape Encoding**: Encodes topological patterns into interpretable shape codes
- **Pattern Library**: Pre-defined topological signatures of truth vs. lies
- **Full Pipeline**: End-to-end from raw text to prediction with explanation
- **Visualization**: Persistence diagrams, barcodes, Betti curves
- **Explainability**: Human-readable explanations of why content is classified as fake

## Dataset

Target: FakeNewsNet (Politifact subset)
Baseline accuracy to beat: 88-93%

## Installation

```bash
cd /mnt/d/VeriLogos
pip install -e .
```

## Usage

```python
from verilogos.applications.lie_detector import LieDetectorPipeline

# Initialize pipeline
pipeline = LieDetectorPipeline()

# Train on dataset
pipeline.train(train_data_path="data/politifact_train.json")

# Predict
result = pipeline.predict("Breaking news: Scientists discover...")

print(f"Prediction: {result['label']}")
print(f"Confidence: {result['confidence']:.2f}")
print(f"Shape Code: {result['shape_code']}")
print(f"Explanation: {result['explanation']}")
```

## Components

### Core
- `truth_geometry.py`: TruthGeometry class for topological analysis
- `shape_encoder.py`: ShapeEncoder for pattern encoding
- `pattern_library.py`: Pre-defined topological signatures

### Data
- `dataset.py`: FakeNewsDataset class
- `loader.py`: Data loading utilities
- `preprocessor.py`: Text preprocessing pipeline

### Topology
- `text_complex_builder.py`: Builds simplicial complexes from text
- `network_complex_builder.py`: Builds complexes from entity networks

### Features
- `persistence_features.py`: Extracts persistence-based features
- `topology_statistics.py`: Computes topological statistics

### Models
- `classifier.py`: ML classifier implementations
- `training.py`: Training loop and optimization

### Evaluation
- `metrics.py`: Evaluation metrics
- `visualization.py`: Plotting and visualization
- `explainability.py`: Explanation generation

## Performance

Target metrics:
- Accuracy: > 90%
- Precision: > 88%
- Recall: > 85%
- F1-Score: > 87%

## License

MIT License - See LICENSE file in repository root
