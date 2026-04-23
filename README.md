# Topological Lie Detector

Topological Lie Detector is a standalone fake-news detection project that combines Topological Data Analysis (TDA) over ACL2017 propagation trees with transformer-based text semantics. The system supports three execution modes (`tda_only`, `text_only`, `hybrid`) and produces reproducible metrics plus publication-style visual artifacts.

## Architecture (Text Diagram)

```text
ACL2017 (twitter15/twitter16)
        |
        +--> Text Path (Tokenizer -> Transformer -> CLS embedding)
        |
        +--> Topology Path (Propagation Tree -> Rigorous TDA Pipeline -> 36D features)
                              |
                              +--> Betti curves / persistence statistics / graph structure metrics

Fusion:
  - tda_only: TDA MLP classifier
  - text_only: Transformer classifier
  - hybrid: Bidirectional Cross-Attention (Text <-> TDA) + fusion head

Outputs:
  - Accuracy/F1 and per-model predictions
  - ROC, PR, confusion matrices, training curves
  - JSON + text reports under ./results
```

## Repository Layout

```text
topological-lie-detector/
├── README.md
├── requirements.txt
├── run.py
├── configs/
│   └── default.yaml
├── src/
│   └── topolie/
│       ├── __init__.py
│       ├── data/
│       │   ├── __init__.py
│       │   └── loaders.py
│       ├── tda/
│       │   ├── __init__.py
│       │   └── pipeline.py
│       ├── models/
│       │   ├── __init__.py
│       │   ├── hybrid.py
│       │   └── trainer.py
│       ├── eval/
│       │   ├── __init__.py
│       │   └── visualizer.py
│       └── experiments/
│           ├── __init__.py
│           └── runner.py
├── checkpoints/
│   └── tda_only/
│       └── best_model.pt
├── data/
│   └── .gitkeep
└── notebooks/
    └── colab_demo.ipynb
```

## Quickstart (Local)

1. Create environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Place ACL2017 dataset at `./data/acl2017` with:

```text
data/acl2017/twitter15/{label.txt,source_tweets.txt,tree/*.txt}
data/acl2017/twitter16/{label.txt,source_tweets.txt,tree/*.txt}
```

3. Run hybrid mode:

```bash
python run.py --mode hybrid --data_path ./data/acl2017
```

## Quickstart (Colab)

Use `notebooks/colab_demo.ipynb`, or run:

```bash
!git clone <YOUR_REPO_URL>
%cd topological-lie-detector
!pip install -r requirements.txt
!python run.py --mode tda_only --data_path ./data/acl2017 --max_events 200 --epochs 5
!python run.py --mode text_only --data_path ./data/acl2017 --max_events 200 --epochs 5
!python run.py --mode hybrid --data_path ./data/acl2017 --max_events 200 --epochs 5
```

## CLI Usage

```bash
python run.py --mode {tda_only,text_only,hybrid} --data_path ./data/acl2017 --max_events 400 --epochs 15
```

Optional:

```bash
python run.py --config configs/default.yaml
```

## Results Table (Placeholder)

| Mode | Accuracy | F1 (Weighted) | Notes |
|------|----------|---------------|-------|
| TDA Only | TBD | TBD | Rigorous topology features only |
| Text Only | TBD | TBD | Transformer semantics only |
| Hybrid | TBD | TBD | Cross-attention fusion |

## Citation (Placeholder)

```text
@article{pourmoslemi2026topological,
  title={Topological Lie Detector: Geometry-Aware Misinformation Detection},
  author={Pourmoslemi, Alireza},
  year={2026}
}
```
