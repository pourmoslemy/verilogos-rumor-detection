"""Compatibility wrapper for trainer module path."""

from src.topolie.models.trainer import EarlyStopping, Trainer, evaluate_model

__all__ = ["EarlyStopping", "Trainer", "evaluate_model"]
