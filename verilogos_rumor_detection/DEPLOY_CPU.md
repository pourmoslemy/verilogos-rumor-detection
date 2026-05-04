# CPU Deployment

## Prerequisites
- Linux x86_64
- Python 3.10+
- Project copied with `data/acl2017` available locally
- Local HuggingFace model/tokenizer directory (offline-safe)

## Setup
```bash
cd topological-lie-detector
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run (CPU, offline-safe)
```bash
python3 run.py \
  --config configs/default.yaml \
  --mode hybrid \
  --data_path ./data/acl2017 \
  --model_path /absolute/path/to/local_model_dir
```

## Optional event cap
```bash
python3 run.py --config configs/default.yaml --max_events 1000 --model_path /absolute/path/to/local_model_dir
```

## Output artifacts
- `results/runtime_config.yaml`
- `results/run.log`
- `results/metrics.json`
- `results/summary.json`
- `checkpoints/<mode>/best_model.pt`
