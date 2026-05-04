"""CLI entrypoint for Topological Lie Detector."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from src.topolie.experiments.runner import run_experiment
from src.topolie.utils import (
    ConfigError,
    ensure_output_paths,
    load_yaml_config,
    save_runtime_config,
    seed_everything,
    setup_logger,
    validate_config,
)

VALID_MODES = {"tda_only", "text_only", "hybrid"}


def _apply_overrides(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    if args.mode is not None:
        config["experiment"]["mode"] = args.mode
    if args.data_path is not None:
        config["data"]["data_path"] = args.data_path
    if args.max_events is not None:
        config["experiment"]["max_events"] = args.max_events
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.model_path is not None:
        config["model"]["text_model_name"] = args.model_path
    return config


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Topological Lie Detector CLI")
    parser.add_argument("--mode", choices=sorted(VALID_MODES), help="Run only one mode.")
    parser.add_argument("--data_path", type=str, help="Path to ACL2017 root containing twitter15/ and twitter16/.")
    parser.add_argument("--max_events", type=int, help="Maximum number of events to use.")
    parser.add_argument("--epochs", type=int, help="Number of training epochs.")
    parser.add_argument("--model_path", type=str, help="Local HuggingFace model/tokenizer path.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="YAML configuration file path.")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    try:
        config = load_yaml_config(args.config)
        config = _apply_overrides(config, args)
        config = validate_config(config)

        paths = ensure_output_paths(config, args.config)
        logger = setup_logger(paths.output_dir)
        seed_everything(int(config["experiment"]["random_seed"]))
        runtime_cfg = save_runtime_config(config, paths.output_dir)

        logger.info("Runtime config saved to %s", runtime_cfg)
        logger.info("Starting experiment mode=%s device=%s", config["experiment"]["mode"], config["experiment"].get("device"))
        summary = run_experiment(config, logger=logger)

        summary_path = paths.output_dir / "summary.json"
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)

        logger.info("Run complete. summary_file=%s", summary_path)
        print("\nRun summary:")
        for model_name, metrics in summary["models"].items():
            print(f"- {model_name}: accuracy={metrics['accuracy']:.4f}, f1_weighted={metrics['f1_weighted']:.4f}")
    except (ConfigError, FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"Run failed: {exc}") from exc


if __name__ == "__main__":
    main()
