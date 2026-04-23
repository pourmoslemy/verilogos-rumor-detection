# 🔬 Topological Lie Detector

**Topological Lie Detector** is a standalone fake-news detection system that combines **Topological Data Analysis (TDA)** on social media propagation trees with **transformer-based text semantics**.

The system analyzes rumor propagation in the **ACL2017 Twitter15 / Twitter16 dataset** and supports multiple modeling strategies to compare **topology-only**, **text-only**, and **hybrid topology–semantic models**.

The project produces **reproducible metrics and publication-quality visual artifacts** for research and experimentation.

---

# 📑 Table of Contents

- Overview
- Architecture
- Repository Structure
- Installation
- Quickstart (Local)
- Quickstart (Colab)
- CLI Usage
- Results
- Citation
- License

---

# 🧠 Overview

Rumor propagation in social networks exhibits **distinct structural patterns** that differ from truthful information diffusion.

This project investigates whether **topological signatures of propagation graphs** can help detect misinformation.

The system combines:

### Text Semantics
Transformer embeddings extracted from tweets.

### Topological Features
Persistent homology features extracted from propagation trees.

### Hybrid Fusion
Cross‑attention fusion of text and topology representations.

---

# 🏗 Architecture
ACL2017 Dataset

(twitter15 / twitter16)

│

│

├─────────────── Text Path ────────────────┐

│ │

│ Tokenizer → Transformer → CLS Embedding

│ │

│

└────────────── Topology Path ─────────────┐

│

Propagation Tree → Rigorous TDA Pipeline → 36D Topology Features

│

│

Betti Curves

Persistence Statistics

Graph Structure Metrics

Fusion Modes:

tda_only

TDA feature vector → MLP classifier

text_only

Transformer CLS embedding → linear classifier

hybrid

Bidirectional cross‑attention between

text embeddings and TDA features

Output:

Predictions

Accuracy / F1

ROC curves

PR curves

Confusion matrices

Training curves

JSON and text reports

📂 Repository Layout
topological-lie-detector/

│

├── README.md

├── requirements.txt

├── run.py

│

├── configs/

│ └── default.yaml

│

├── src/

│ └── topolie/

│ │

│ ├── init.py

│ │

│ ├── data/

│ │ ├── init.py

│ │ └── loaders.py

│ │

│ ├── tda/

│ │ ├── init.py

│ │ └── pipeline.py

│ │

│ ├── models/

│ │ ├── init.py

│ │ ├── hybrid.py

│ │ └── trainer.py

│ │

│ ├── eval/

│ │ ├── init.py

│ │ └── visualizer.py

│ │

│ └── experiments/

│ ├── init.py

│ └── runner.py

│

├── checkpoints/

│ └── tda_only/

│ └── best_model.pt

│

├── data/

│ └── .gitkeep

│

└── notebooks/

└── colab_demo.ipynb

⚙ Installation
Create a Python environment and install dependencies.

python -m venv .venv

Activate environment:

Windows

.venv\Scripts\activate

Linux / Mac

source .venv/bin/activate

Install dependencies

pip install -r requirements.txt

🚀 Quickstart (Local)
Place the ACL2017 dataset in the following structure:

data/acl2017/

twitter15/

label.txt

source_tweets.txt

tree/

twitter16/

label.txt

source_tweets.txt

tree/

Run the hybrid model:

python run.py --mode hybrid --data_path ./data/acl2017

☁️ Quickstart (Google Colab)
Clone the repository:

!git clone https://github.com/YOUR_USERNAME/topological-lie-detector

%cd topological-lie-detector

Install dependencies:

!pip install -r requirements.txt

Run the three modes:

!python run.py --mode tda_only --data_path ./data/acl2017 --max_events 200 --epochs 5

!python run.py --mode text_only --data_path ./data/acl2017 --max_events 200 --epochs 5

!python run.py --mode hybrid --data_path ./data/acl2017 --max_events 200 --epochs 5

💻 CLI Usage
Basic command:

python run.py \

–mode hybrid \

–data_path ./data/acl2017 \

–max_events 400 \

–epochs 15

Available modes:

tda_only

text_only

hybrid

Optional configuration file:

python run.py --config configs/default.yaml

📊 Results (Placeholder)
Mode	Accuracy	Weighted F1	Notes
TDA Only	TBD	TBD	Topological features only
Text Only	TBD	TBD	Transformer text features
Hybrid	TBD	TBD	Cross‑attention fusion
📄 Citation
If you use this work in research, please cite:

@article{pourmoslemi2026topological,

title={Topological Lie Detector: Geometry-Aware Misinformation Detection},

author={Pourmoslemi, Alireza},

year={2026}

}

📜 License
MIT License
