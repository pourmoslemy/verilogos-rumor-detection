#!/usr/bin/env python3
"""Quick test of rigorous TDA pipeline on 100 events."""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
import warnings
warnings.filterwarnings('ignore')

from rigorous_tda_pipeline import RigorousTDAPipeline

def load_sample_events(base_path: str, max_events: int = 100):
    """Load a sample of events for quick testing."""
    print(f"Loading {max_events} sample events...")
    
    all_data = []
    
    for dataset_name in ['twitter15', 'twitter16']:
        dataset_path = Path(base_path) / dataset_name
        
        # Load labels
        label_file = dataset_path / 'label.txt'
        labels = {}
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split(':')
                if len(parts) != 2:
                    continue
                label, tweet_id = parts
                labels[tweet_id] = 1 if label == 'true' else 0
        
        # Load trees
        tree_dir = dataset_path / 'tree'
        count = 0
        for tree_file in tree_dir.glob('*.txt'):
            if count >= max_events // 2:
                break
            
            tweet_id = tree_file.stem
            if tweet_id not in labels:
                continue
            
            all_data.append({
                'event_id': f"{dataset_name}_{tweet_id}",
                'label': labels[tweet_id],
                'tree_file': tree_file
            })
            count += 1
    
    print(f"Loaded {len(all_data)} events")
    print(f"  Real: {sum(1 for d in all_data if d['label'] == 1)}")
    print(f"  Fake: {sum(1 for d in all_data if d['label'] == 0)}")
    
    return all_data

def main():
    print("="*80)
    print("QUICK TDA TEST (100 events)")
    print("="*80)
    
    data_path = '/mnt/d/Verilogos/historical_data/rumor_detection_acl2017'
    
    # Load sample
    events = load_sample_events(data_path, max_events=100)
    
    # Extract features
    print("\nExtracting TDA features...")
    pipeline = RigorousTDAPipeline(lambda_decay=0.001, temporal_window=60.0)
    
    features_list = []
    labels_list = []
    
    for i, event in enumerate(events):
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(events)}...")
        
        try:
            features, metadata = pipeline.process_cascade(event['tree_file'])
            features_list.append(features)
            labels_list.append(event['label'])
        except Exception as e:
            print(f"  Error on {event['event_id']}: {e}")
            continue
    
    print(f"\nExtracted features for {len(features_list)} events")
    
    # Train/test split
    X = np.array(features_list)
    y = np.array(labels_list)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Normalize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    print("\nTraining models...")
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced'),
        'Random Forest': RandomForestClassifier(n_estimators=100, class_weight='balanced')
    }
    
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print(f"\n{name}:")
        print(f"  F1 Score: {f1:.4f}")
        print(classification_report(y_test, y_pred, target_names=['Fake', 'Real']))

if __name__ == '__main__':
    main()
