#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODEL_PATH="${MODEL_PATH:-}"
if [[ -z "$MODEL_PATH" ]]; then
  echo "ERROR: set MODEL_PATH to a local HuggingFace model/tokenizer directory"
  echo "Example: MODEL_PATH=/opt/models/distilroberta-base-local scripts/run_cpu.sh"
  exit 1
fi

python3 run.py \
  --config configs/default.yaml \
  --mode hybrid \
  --data_path ./data/acl2017 \
  --model_path "$MODEL_PATH"
