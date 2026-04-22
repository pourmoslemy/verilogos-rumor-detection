#!/usr/bin/env python3
"""Test current system to validate bug fixes."""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from verilogos.experiments.datasets import FakeNewsNetDataset
from verilogos.experiments.models.classical import (
    LogisticRegressionModel, RandomForestModel, XGBoostModel
)
from verilogos.experiments.features import TextFeatureExtractor
from sklearn.metrics import f1_score, accuracy_score

print("Testing Current System - Validating Bug Fixes")
print("="*80)

# Load dataset
dataset = FakeNewsNetDataset()
dataset.create_synthetic(n_samples=300)
print(f"✓ Created {len(dataset)} samples")

# Stratified splits
train_data, val_data, test_data = dataset.get_splits()
print(f"✓ Splits: {len(train_data)} train, {len(test_data)} test")

# Extract features (testing leakage fix)
train_texts = [s.text for s in train_data]
test_texts = [s.text for s in test_data]
train_labels = np.array([s.label for s in train_data])
test_labels = np.array([s.label for s in test_data])

extractor = TextFeatureExtractor(max_features=500)
X_train = extractor.fit_transform(train_texts)
X_test = extractor.transform(test_texts)
print(f"✓ Features: {X_train.shape}")

# Train models
models = {
    'LogisticRegression': LogisticRegressionModel(),
    'RandomForest': RandomForestModel(n_estimators=100),
    'XGBoost': XGBoostModel(n_estimators=100)
}

print("\nResults:")
for name, model in models.items():
    model.fit(X_train, train_labels)
    y_pred = model.predict(X_test)
    f1 = f1_score(test_labels, y_pred, average='macro')
    acc = accuracy_score(test_labels, y_pred)
    print(f"  {name}: F1={f1:.4f}, Acc={acc:.4f}")

print("\n✓ If F1 ≥ 0.70, proceed to Twitter15/16 integration")
