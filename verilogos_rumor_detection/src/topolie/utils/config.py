from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping

import yaml


class ConfigError(ValueError):
    """Raised when runtime config is missing required keys or values."""


@dataclass(frozen=True)
class Paths:
    config_path: Path
    output_dir: Path
    checkpoint_dir: Path


def _require(mapping: Mapping[str, Any], key: str, ctx: str) -> Any:
    if key not in mapping:
        raise ConfigError(f"Missing required key '{ctx}.{key}'")
    return mapping[key]


def load_yaml_config(config_path: str | Path) -> Dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    if not isinstance(cfg, dict):
        raise ConfigError("Top-level config must be a mapping")
    return cfg


def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    required_sections = ["experiment", "data", "split", "tda", "model", "training", "output"]
    for section in required_sections:
        _require(config, section, "config")
        if not isinstance(config[section], dict):
            raise ConfigError(f"Section '{section}' must be a mapping")

    exp = config["experiment"]
    split = config["split"]
    train = config["training"]
    out = config["output"]
    model = config["model"]

    if exp.get("mode") is None:
        raise ConfigError("experiment.mode is required")
    if exp.get("random_seed") is None:
        raise ConfigError("experiment.random_seed is required")

    ratios = [float(split.get("train_ratio", 0.0)), float(split.get("val_ratio", 0.0)), float(split.get("test_ratio", 0.0))]
    if any(r <= 0 for r in ratios):
        raise ConfigError("split ratios must be positive")
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ConfigError("split ratios must sum to 1.0")

    if int(train.get("batch_size", 0)) <= 0:
        raise ConfigError("training.batch_size must be > 0")
    if int(train.get("epochs", 0)) <= 0:
        raise ConfigError("training.epochs must be > 0")

    if not out.get("results_dir") or not out.get("checkpoint_dir"):
        raise ConfigError("output.results_dir and output.checkpoint_dir are required")

    if not model.get("text_model_name"):
        raise ConfigError("model.text_model_name is required")

    max_events = exp.get("max_events")
    if max_events is not None and int(max_events) <= 0:
        raise ConfigError("experiment.max_events must be > 0 when provided")

    return config


def ensure_output_paths(config: Dict[str, Any], config_path: str | Path) -> Paths:
    output_dir = Path(config["output"]["results_dir"]).resolve()
    checkpoint_dir = Path(config["output"]["checkpoint_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return Paths(config_path=Path(config_path).resolve(), output_dir=output_dir, checkpoint_dir=checkpoint_dir)


def save_runtime_config(config: Dict[str, Any], output_dir: Path) -> Path:
    destination = output_dir / "runtime_config.yaml"
    with destination.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    return destination
