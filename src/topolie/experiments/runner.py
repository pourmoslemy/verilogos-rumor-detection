"""
Experiment runner for Topological Lie Detector.

This module exposes a callable `run_experiment(config)` function used by `run.py`.
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import f1_score

from src.topolie.data.loaders import (
    create_dataloaders,
    extract_tda_features_parallel,
    load_acl2017_dataset,
)
from src.topolie.eval.visualizer import generate_all_visualizations
from src.topolie.models.hybrid import create_model
from src.topolie.models.trainer import Trainer, evaluate_model


def _normalize_modes(mode_value: Any) -> List[str]:
    if isinstance(mode_value, str):
        return [mode_value]
    if isinstance(mode_value, list):
        return mode_value
    raise ValueError("`mode` must be a string or list of strings")


def _train_single_mode(
    mode: str,
    train_loader,
    val_loader,
    test_loader,
    config: Dict[str, Any],
    device: str,
) -> Tuple[Dict[str, Any], Dict[str, List[float]]]:
    print("\n" + "=" * 80)
    print(f"TRAINING {mode.upper()} MODEL")
    print("=" * 80)

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
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
        num_epochs=train_cfg["epochs"],
        patience=train_cfg["patience"],
        checkpoint_dir=f"{output_cfg['checkpoint_dir']}/{mode}",
    )

    history = trainer.train()

    print(f"\nEvaluating {mode.upper()} model on test set...")
    test_results = evaluate_model(model, test_loader, device)
    print(f"Test Accuracy: {test_results['accuracy']:.4f}")

    return test_results, history


def run_experiment(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run end-to-end experiment based on configuration.

    Args:
        config: Configuration dictionary loaded from YAML + CLI overrides.

    Returns:
        Summary dictionary containing per-model metrics and artifact paths.
    """
    mode_list = _normalize_modes(config["experiment"]["mode"])

    if config["experiment"]["device"] == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config["experiment"]["device"]

    print(f"Using device: {device}")

    output_dir = Path(config["output"]["results_dir"])
    checkpoint_dir = Path(config["output"]["checkpoint_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    data_path = config["data"]["data_path"]
    if not Path(data_path).exists():
        raise FileNotFoundError(
            f"Dataset path not found: {data_path}. "
            "Provide ACL2017 root containing twitter15/ and twitter16/."
        )

    print("\n" + "=" * 80)
    print("STEP 1: LOADING DATASET")
    print("=" * 80)
    events, source_tweets = load_acl2017_dataset(
        base_path=data_path,
        max_events=config["experiment"]["max_events"],
        balance_classes=config["data"]["balance_classes"],
    )

    precomputed_tda = None
    if any(mode in mode_list for mode in ["tda_only", "hybrid"]):
        print("\n" + "=" * 80)
        print("STEP 2: EXTRACTING TDA FEATURES")
        print("=" * 80)
        precomputed_tda = extract_tda_features_parallel(
            events=events,
            n_workers=config["tda"]["n_workers"],
            lambda_decay=config["tda"]["lambda_decay"],
            temporal_window=config["tda"]["temporal_window"],
        )
        tda_save_path = output_dir / "tda_features.npy"
        np.save(tda_save_path, precomputed_tda)
        print(f"Saved TDA features to {tda_save_path}")

    all_results: Dict[str, Dict[str, Any]] = {}
    all_histories: Dict[str, Dict[str, List[float]]] = {}

    print("\n" + "=" * 80)
    print("STEP 3: TRAIN AND EVALUATE")
    print("=" * 80)

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
            num_workers=config["training"]["num_workers"],
        )

        test_results, history = _train_single_mode(
            mode=mode,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            config=config,
            device=device,
        )

        model_name = mode.replace("_", " ").title()
        all_results[model_name] = test_results
        all_histories[model_name] = history

    print("\n" + "=" * 80)
    print("STEP 4: GENERATING VISUALIZATIONS")
    print("=" * 80)
    generate_all_visualizations(
        results=all_results,
        training_histories=all_histories,
        output_dir=str(output_dir),
    )

    summary: Dict[str, Any] = {"models": {}, "artifacts": {}}

    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    for model_name, result in all_results.items():
        weighted_f1 = f1_score(result["labels"], result["predictions"], average="weighted")
        summary["models"][model_name] = {
            "accuracy": float(result["accuracy"]),
            "f1_weighted": float(weighted_f1),
        }
        print(f"\n{model_name}:")
        print(f"  Accuracy: {result['accuracy']:.4f}")
        print(f"  F1 Score: {weighted_f1:.4f}")

    summary["artifacts"] = {
        "results_dir": str(output_dir),
        "checkpoint_dir": str(checkpoint_dir),
    }

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)
    print(f"Results saved to: {output_dir}")
    print(f"Checkpoints saved to: {checkpoint_dir}")

    return summary
