# Topological Lie Detector - Quick Start Guide

## Installation

Ensure VeriLogos is installed and PYTHONPATH is set:

```bash
cd /mnt/d/VeriLogos
export PYTHONPATH=/mnt/d/VeriLogos:$PYTHONPATH
```

## Run Your First Experiment (30 seconds)

```bash
# Run with synthetic data (100 samples)
python3 -m verilogos.applications.lie_detector.run_experiment

# Output will be in ./results/ directory
```

## What You'll Get

```
results/
├── experiment_results.json       # Metrics: accuracy, precision, recall, F1
├── lie_detector_model.pkl        # Trained model (can be reused)
├── confusion_matrix.png          # Visual performance matrix
├── roc_curve.png                 # ROC curve
└── persistence_diagram_*.png     # Topological visualizations
```

## Example Output

```
================================================================================
TOPOLOGICAL LIE DETECTOR - EXPERIMENT
================================================================================

[1/6] Loading dataset...
Generated 100 articles (50 real, 50 fake)

[2/6] Splitting train/test sets...
Split: 80 train, 20 test

[3/6] Initializing pipeline...

[4/6] Training pipeline...
Processing article 10/80
Processing article 20/80
...
Training complete!

[5/6] Evaluating on test set...

================================================================================
EVALUATION RESULTS
================================================================================
Accuracy:  0.9500
Precision: 0.9474
Recall:    0.9000
F1 Score:  0.9231
ROC-AUC:   0.9750
================================================================================

[6/6] Generating visualizations...
Saved confusion matrix to results/confusion_matrix.png
Saved ROC curve to results/roc_curve.png

--- Example 1 ---
True Label: fake
Prediction: fake
Confidence: 0.9200
Shape Code: multi_loop_fragile_burst
Explanation: High fragmentation suggests disconnected narrative structure...

EXPERIMENT COMPLETE
```

## Use in Your Code

```python
from verilogos.applications.lie_detector.pipeline import LieDetectorPipeline

# Initialize
pipeline = LieDetectorPipeline()

# Train
texts = ["Real news article...", "Fake news article..."]
labels = [0, 1]  # 0=real, 1=fake
pipeline.train(texts, labels)

# Predict
result = pipeline.predict("New article to check...")
print(f"{result['label']}: {result['confidence']:.1%} confident")
print(f"Shape: {result['shape_code']}")
```

## Advanced Options

```bash
# Use your own dataset
python3 -m verilogos.applications.lie_detector.run_experiment \
    --data-path /path/to/fakenewsnet.json \
    --n-samples 500 \
    --test-size 0.3

# Fast mode (no visualizations)
python3 -m verilogos.applications.lie_detector.run_experiment \
    --no-visualize \
    --output-dir quick_test

# Reproducible results
python3 -m verilogos.applications.lie_detector.run_experiment \
    --random-seed 123
```

## Understanding Results

### Shape Codes
- `truth_stable`: Coherent, well-connected narrative
- `narrative_loop`: Circular reasoning detected
- `propaganda_cloud`: Dense, repetitive structure
- `rumor_burst`: Fragile, short-lived features
- `collapse_pattern`: Rapid decay of structure

### Topological Features
- **B0 (Betti-0)**: Number of connected components
- **B1 (Betti-1)**: Number of loops (circular reasoning)
- **B2 (Betti-2)**: Number of voids (structural holes)
- **Fragmentation**: How disconnected the narrative is
- **Coherence**: How well claims connect together
- **Stability**: How persistent topological features are

## Troubleshooting

**Import Error:**
```bash
export PYTHONPATH=/mnt/d/VeriLogos:$PYTHONPATH
```

**No sklearn:**
```bash
pip install scikit-learn numpy
```

**No matplotlib (optional):**
```bash
pip install matplotlib
# Or run with --no-visualize
```

## Next Steps

1. ✅ Run default experiment
2. ✅ Check results in `./results/`
3. ✅ Try with your own text data
4. ✅ Integrate into your application
5. ✅ Tune hyperparameters for your use case

---

**Ready to detect misinformation with topology!** 🔍📊
