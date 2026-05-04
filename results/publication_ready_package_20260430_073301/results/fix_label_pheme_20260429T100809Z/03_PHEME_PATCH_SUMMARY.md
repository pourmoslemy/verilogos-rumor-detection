# PHEME Patch Summary

## Files modified
- `/home/ubuntu/colab_deployment/src/topolie/data/loaders.py`
- `/home/ubuntu/colab_deployment/src/topolie/experiments/runner.py`
- `/home/ubuntu/colab_deployment/src/topolie/tda/pipeline.py`
- `/home/ubuntu/colab_deployment/run.py`

## Dataset dispatch
- `src/topolie/data/loaders.py` now exposes `load_dataset(data_path, ...)`.
- Dispatch rules:
  - ACL2017 root or direct `twitter15` / `twitter16` directory -> `load_acl2017_dataset()`
  - processed PHEME directory with `label.txt`, `source_tweets.txt`, `tree/`, and `root_ids.txt` / `metadata.jsonl` -> `load_pheme_dataset()`
- `src/topolie/experiments/runner.py` now calls `load_dataset(...)` instead of hardcoding `load_acl2017_dataset(...)`.

## PHEME path resolution
- The rebuilt dataset path expected by the new loader is:
  `/home/ubuntu/colab_deployment/data/pheme/processed_ready_20260429`
- `run.py --data_path` help text was widened so the current CLI can point either to ACL2017 or to a processed PHEME directory.

## Expected input format
- `label.txt`
- `source_tweets.txt`
- `root_ids.txt`
- `metadata.jsonl`
- `tree/<root_id>.txt`
- `splits/train_events.txt`
- `splits/val_events.txt`
- `splits/test_events.txt`

## Validation added
- dataset path existence checks
- required-file checks
- strict binary label normalization (`REAL=0`, `FAKE=1`)
- duplicate-root detection
- tree-file validation
- split overlap detection across train/val/test
- source-text presence checks

## Compatibility notes
- twitter15/twitter16 stay on the same loader path and preserve their layout.
- PHEME is added via a narrow adapter path instead of a broad refactor.
- `src/topolie/tda/pipeline.py` gained tab-delimited tree parsing so PHEME trees can flow through the current TDA feature extractor without changing dataset files.
