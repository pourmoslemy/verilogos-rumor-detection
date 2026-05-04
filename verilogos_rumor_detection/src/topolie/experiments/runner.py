"""Experiment runner for Topological Lie Detector."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import f1_score

from src.topolie.data.loaders import create_dataloaders, extract_tda_features_parallel, load_acl2017_dataset
from src.topolie.eval.visualizer import generate_all_visualizations
from src.topolie.models.hybrid import create_model
from src.topolie.trainers.trainer import Trainer, evaluate_model


def _normalize_modes(mode_value: Any) -> List[str]:
    if isinstance(mode_value, str):
        return [mode_value]
    if isinstance(mode_value, list):
        return mode_value
    raise ValueError("experiment.mode must be a string or list of strings")


def _train_single_mode(
    mode: str,
    train_loader,
    val_loader,
    test_loader,
    config: Dict[str, Any],
    device: str,
    logger: logging.Logger,
) -> Tuple[Dict[str, Any], Dict[str, List[float]]]:
    logger.info("Training mode=%s", mode)

    model_cfg = config["model"]
    train_cfg = config["training"]
    output_cfg = config["output"]

    model = create_model(
        mode=mode,
        text_model_name=model_cfg["text_model_name"],
        device=device,
        tda_input_dim=model_cfg["tda_input_dim"],
        tda_hidden_dims=model_cfg["tda_hidden_dims"],
        embed_dim=model_cfg["embed_dim"],
        num_attention_heads=model_cfg["num_attention_heads"],
        dropout=model_cfg["dropout"],
        freeze_text_encoder=model_cfg["freeze_text_encoder"],
        local_files_only=bool(model_cfg.get("local_files_only", True)),
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
        num_epochs=train_cfg["epochs"],
        patience=int(train_cfg.get("early_stopping_patience", train_cfg["patience"])),
        explicit_class_weights=config["data"].get("explicit_class_weights"),
        checkpoint_dir=f"{output_cfg['checkpoint_dir']}/{mode}",
    )

    history = trainer.train()

    logger.info("Evaluating mode=%s on test split", mode)
    test_results = evaluate_model(model, test_loader, device)
    logger.info("mode=%s test_accuracy=%.4f", mode, test_results["accuracy"])

    return test_results, history


def run_experiment(config: Dict[str, Any], logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
    logger = logger or logging.getLogger("topolie")
    mode_list = _normalize_modes(config["experiment"]["mode"])

    if config["experiment"].get("device") == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = str(config["experiment"]["device"])

    if device != "cpu" and not torch.cuda.is_available():
        logger.warning("Requested device=%s but CUDA is unavailable. Falling back to cpu.", device)
        device = "cpu"

    logger.info("Using device=%s", device)

    output_dir = Path(config["output"]["results_dir"])
    checkpoint_dir = Path(config["output"]["checkpoint_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    data_path = config["data"]["data_path"]
    if not Path(data_path).exists():
        raise FileNotFoundError(
            f"Dataset path not found: {data_path}. Provide ACL2017 root containing twitter15/ and twitter16/."
        )

    logger.info("Loading ACL2017 dataset from %s", data_path)
    events, source_tweets, load_stats = load_acl2017_dataset(
        base_path=data_path,
        max_events=config["experiment"].get("max_events"),
        balance_classes=bool(config["data"].get("balance_classes", True)),
        random_seed=int(config["experiment"]["random_seed"]),
    )
    logger.info("Dataset load stats: %s", json.dumps(load_stats, sort_keys=True))

    precomputed_tda = None
    if any(mode in mode_list for mode in ["tda_only", "hybrid"]):
        logger.info("Extracting TDA features for %d events", len(events))
        precomputed_tda = extract_tda_features_parallel(
            events=events,
            n_workers=config["tda"]["n_workers"],
            lambda_decay=config["tda"]["lambda_decay"],
            temporal_window=config["tda"]["temporal_window"],
        )
        tda_save_path = output_dir / "tda_features.npy"
        np.save(tda_save_path, precomputed_tda)
        logger.info("Saved TDA features to %s", tda_save_path)

    all_results: Dict[str, Dict[str, Any]] = {}
    all_histories: Dict[str, Dict[str, List[float]]] = {}

    for mode in mode_list:
        train_loader, val_loader, test_loader = create_dataloaders(
            events=events,
            source_tweets=source_tweets,
            mode=mode,
            batch_size=config["training"]["batch_size"],
            train_ratio=config["split"]["train_ratio"],
            val_ratio=config["split"]["val_ratio"],
            test_ratio=config["split"]["test_ratio"],
            random_seed=config["experiment"]["random_seed"],
            precomputed_tda=precomputed_tda,
            tokenizer_name=config["model"]["text_model_name"],
            local_files_only=bool(config["model"].get("local_files_only", True)),
            balance_classes=bool(config["data"].get("balance_classes", False)),
            num_workers=config["training"]["num_workers"],
        )

        test_results, history = _train_single_mode(
            mode=mode,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            config=config,
            device=device,
            logger=logger,
        )

        model_name = mode.replace("_", " ").title()
        all_results[model_name] = test_results
        all_histories[model_name] = history

    generate_all_visualizations(results=all_results, training_histories=all_histories, output_dir=str(output_dir))

    summary: Dict[str, Any] = {"models": {}, "artifacts": {}}
    for model_name, result in all_results.items():
        weighted_f1 = f1_score(result["labels"], result["predictions"], average="weighted")
        summary["models"][model_name] = {"accuracy": float(result["accuracy"]), "f1_weighted": float(weighted_f1)}

    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(summary["models"], handle, indent=2)

    summary["artifacts"] = {
        "results_dir": str(output_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "metrics_file": str(metrics_path),
        "load_stats": load_stats,
    }

    logger.info("Experiment complete. results_dir=%s checkpoint_dir=%s", output_dir, checkpoint_dir)
    return summary
