"""CLI entrypoint for Topological Lie Detector."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import yaml

from src.topolie.experiments.runner import run_experiment


VALID_MODES = {"tda_only", "text_only", "hybrid"}


def _load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _apply_overrides(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    if args.mode is not None:
        config["experiment"]["mode"] = args.mode
    if args.data_path is not None:
        config["data"]["data_path"] = args.data_path
    if args.max_events is not None:
        config["experiment"]["max_events"] = args.max_events
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    return config


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Topological Lie Detector CLI")
    parser.add_argument(
        "--mode",
        choices=sorted(VALID_MODES),
        help="Run only one mode. If omitted, mode from config is used.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        help="Path to an ACL2017 root (twitter15/twitter16) or a processed PHEME directory.",
    )
    parser.add_argument(
        "--max_events",
        type=int,
        help="Maximum number of events to use.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="YAML configuration file path.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config = _load_config(str(config_path))
    config = _apply_overrides(config, args)

    summary = run_experiment(config)
    print("\nRun summary:")
    for model_name, metrics in summary["models"].items():
        print(
            f"- {model_name}: accuracy={metrics['accuracy']:.4f}, "
            f"f1_weighted={metrics['f1_weighted']:.4f}"
        )


if __name__ == "__main__":
    main()
