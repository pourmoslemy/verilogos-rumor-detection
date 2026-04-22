#!/usr/bin/env python3
"""Test pipeline on a single event."""

from pathlib import Path
from rigorous_tda_pipeline import RigorousTDAPipeline

# Get first tree file
tree_file = Path('/mnt/d/Verilogos/historical_data/rumor_detection_acl2017/twitter15/tree')
first_file = list(tree_file.glob('*.txt'))[0]

print(f"Testing on: {first_file}")

pipeline = RigorousTDAPipeline(lambda_decay=0.001, temporal_window=60.0)

try:
    features, metadata = pipeline.process_cascade(first_file)
    print(f"\nSuccess!")
    print(f"Features shape: {features.shape}")
    print(f"Metadata: {metadata}")
    print(f"\nFirst 10 features: {features[:10]}")
except Exception as e:
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()
