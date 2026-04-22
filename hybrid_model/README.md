# Hybrid TDA-Text Neural Network for Fake News Detection

State-of-the-art hybrid architecture combining Topological Data Analysis (TDA) and Transformer-based text embeddings with Cross-Attention fusion for fake news detection.

## Architecture

### Three Independent Modalities

1. **TDA-Only Model**
   - Input: 36-dimensional topological features (Betti curves, persistence landscapes, graph topology, temporal dynamics)
   - Architecture: MLP projection (36 → 128 → 256 → 512 → 768) with BatchNorm and Dropout
   - Output: Binary classification (Fake/Real)

2. **Text-Only Model**
   - Input: Tweet text
   - Architecture: DistilRoBERTa-base encoder → [CLS] embedding (768-dim)
   - Output: Binary classification (Fake/Real)

3. **Hybrid Model (Cross-Attention Fusion)**
   - **Text Pathway**: DistilRoBERTa → [CLS] embedding (768-dim)
   - **TDA Pathway**: 36D features → MLP → 768-dim projection
   - **Fusion Layer**: Bidirectional Multi-Head Cross-Attention
     - Text attends to TDA (Text as Query, TDA as Key/Value)
     - TDA attends to Text (TDA as Query, Text as Key/Value)
     - Concatenate attended representations (1536-dim)
   - **Classification Head**: Fused embedding → FC layers → Binary output

## Features

- **Modular Design**: Train and evaluate each modality independently
- **Parallel TDA Extraction**: Multiprocessing support for fast feature extraction
- **Early Stopping**: Prevents overfitting with patience-based stopping
- **Learning Rate Scheduling**: ReduceLROnPlateau for adaptive learning
- **Publication-Quality Visualizations**:
  - ROC curves (all models on same plot)
  - Precision-Recall curves
  - Confusion matrices
  - Metrics comparison bar charts
  - Training history plots

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Quick Start (400 events, ~30 minutes)

```bash
cd /mnt/d/Verilogos/hybrid_model
python main_experiment.py \
    --max_events 400 \
    --batch_size 16 \
    --num_epochs 15 \
    --n_workers_tda 4 \
    --modes tda_only text_only hybrid
```

### Full Dataset (2308 events, ~4 hours)

```bash
python main_experiment.py \
    --max_events 2308 \
    --batch_size 32 \
    --num_epochs 20 \
    --n_workers_tda 8
```

### Train Single Model

```bash
# TDA-only
python main_experiment.py --modes tda_only --max_events 400

# Text-only
python main_experiment.py --modes text_only --max_events 400

# Hybrid
python main_experiment.py --modes hybrid --max_events 400
```

## Command-Line Arguments

- `--data_path`: Path to ACL2017 dataset (default: `/mnt/d/Verilogos/historical_data/rumor_detection_acl2017`)
- `--max_events`: Maximum number of events (default: 400, use None for all)
- `--batch_size`: Training batch size (default: 16)
- `--num_epochs`: Number of training epochs (default: 15)
- `--learning_rate`: Learning rate (default: 2e-5)
- `--n_workers_tda`: Parallel workers for TDA extraction (default: 4)
- `--output_dir`: Output directory for results (default: `./results`)
- `--checkpoint_dir`: Directory for model checkpoints (default: `./checkpoints`)
- `--modes`: Which models to train (default: all three)

## Output Structure

```
results/
├── roc_curves.png                    # ROC curves for all models
├── pr_curves.png                     # Precision-Recall curves
├── confusion_matrix_tda_only.png     # Confusion matrix (TDA)
├── confusion_matrix_text_only.png    # Confusion matrix (Text)
├── confusion_matrix_hybrid.png       # Confusion matrix (Hybrid)
├── metrics_comparison.png            # Bar chart comparing all metrics
├── training_history_tda_only.png     # Training curves (TDA)
├── training_history_text_only.png    # Training curves (Text)
├── training_history_hybrid.png       # Training curves (Hybrid)
├── classification_report.txt         # Detailed metrics for all models
├── results.json                      # Numerical results in JSON format
└── tda_features.npy                  # Precomputed TDA features

checkpoints/
├── tda_only/
│   ├── best_model.pt
│   └── latest_checkpoint.pt
├── text_only/
│   ├── best_model.pt
│   └── latest_checkpoint.pt
└── hybrid/
    ├── best_model.pt
    └── latest_checkpoint.pt
```

## Expected Performance (400 events)

Based on previous experiments:

| Model | F1 Score | Accuracy | Notes |
|-------|----------|----------|-------|
| TDA-Only | ~0.66 | ~0.67 | Pure topology signal |
| Text-Only | ~0.90 | ~0.90 | DistilRoBERTa baseline |
| **Hybrid** | **~0.92** | **~0.92** | **+2% improvement** |

## Architecture Details

### TDA Projection Network

```
Input: (batch_size, 36)
  ↓
Linear(36 → 128) + BatchNorm + ReLU + Dropout(0.3)
  ↓
Linear(128 → 256) + BatchNorm + ReLU + Dropout(0.3)
  ↓
Linear(256 → 512) + BatchNorm + ReLU + Dropout(0.3)
  ↓
Linear(512 → 768)
  ↓
Output: (batch_size, 768)
```

### Multi-Head Cross-Attention

```
Query: (batch_size, 768)
Key:   (batch_size, 768)
Value: (batch_size, 768)
  ↓
Q_proj, K_proj, V_proj → (batch_size, num_heads, head_dim)
  ↓
Attention Scores = softmax(Q·K^T / √d_k)
  ↓
Attended = Attention_Scores · V
  ↓
Concatenate heads → (batch_size, 768)
  ↓
Output projection
```

### Fusion Strategy

```
Text Embedding (768) ──┐
                       ├─→ Cross-Attention ─→ Text_attended (768)
TDA Embedding (768) ───┘

TDA Embedding (768) ───┐
                       ├─→ Cross-Attention ─→ TDA_attended (768)
Text Embedding (768) ──┘

Concatenate [Text_attended, TDA_attended] → (1536)
  ↓
Linear(1536 → 768) + LayerNorm + ReLU + Dropout
  ↓
Fused Embedding (768)
  ↓
Classifier: Linear(768 → 256) + ReLU + Dropout + Linear(256 → 2)
```

## Key Implementation Details

1. **Stratified Splitting**: Ensures balanced class distribution in train/val/test sets
2. **Class Weighting**: Handles class imbalance with `class_weight='balanced'`
3. **Gradient Clipping**: Prevents exploding gradients (max_norm=1.0)
4. **Mixed Precision**: Can be enabled for faster training on modern GPUs
5. **Reproducibility**: Fixed random seeds (42) for consistent results

## Citation

If you use this code in your research, please cite:

```bibtex
@article{pourmoslemi2025geometry,
  title={The Geometry of Truth: Topological Data Analysis for Fake News Detection},
  author={Pourmoslemi, Alireza},
  journal={arXiv preprint},
  year={2025}
}
```

## License

MIT License - See LICENSE file for details

## Author

Alireza Pourmoslemi (apmath99@gmail.com)

## Acknowledgments

- ACL2017 Rumor Detection Dataset
- HuggingFace Transformers
- PyTorch Team
