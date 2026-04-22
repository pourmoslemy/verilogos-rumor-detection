#!/usr/bin/env python3
"""Balanced test: 100 real + 100 fake events."""

import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

from rigorous_tda_pipeline import RigorousTDAPipeline, SCLogicRuleEngine

def load_balanced_events(base_path: str, n_per_class: int = 100):
    """Load balanced sample of real and fake events."""
    print(f"Loading {n_per_class} real + {n_per_class} fake events...")
    
    real_events = []
    fake_events = []
    
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
        for tree_file in tree_dir.glob('*.txt'):
            tweet_id = tree_file.stem
            if tweet_id not in labels:
                continue
            
            event = {
                'event_id': f"{dataset_name}_{tweet_id}",
                'label': labels[tweet_id],
                'tree_file': tree_file
            }
            
            if labels[tweet_id] == 1 and len(real_events) < n_per_class:
                real_events.append(event)
            elif labels[tweet_id] == 0 and len(fake_events) < n_per_class:
                fake_events.append(event)
            
            if len(real_events) >= n_per_class and len(fake_events) >= n_per_class:
                break
        
        if len(real_events) >= n_per_class and len(fake_events) >= n_per_class:
            break
    
    all_events = real_events + fake_events
    print(f"Loaded {len(all_events)} events: {len(real_events)} real, {len(fake_events)} fake")
    
    return all_events

def main():
    print("="*80)
    print("BALANCED TDA TEST (200 events: 100 real + 100 fake)")
    print("="*80)
    
    data_path = '/mnt/d/Verilogos/historical_data/rumor_detection_acl2017'
    
    # Load balanced sample
    events = load_balanced_events(data_path, n_per_class=100)
    
    # Extract features
    print("\nExtracting TDA features...")
    pipeline = RigorousTDAPipeline(lambda_decay=0.001, temporal_window=60.0)
    
    features_list = []
    labels_list = []
    event_ids = []
    
    for i, event in enumerate(events):
        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{len(events)}...")
        
        try:
            features, metadata = pipeline.process_cascade(event['tree_file'])
            features_list.append(features)
            labels_list.append(event['label'])
            event_ids.append(event['event_id'])
        except Exception as e:
            print(f"  Error on {event['event_id']}: {e}")
            continue
    
    print(f"\nExtracted features for {len(features_list)} events")
    
    # Convert to arrays
    X = np.array(features_list)
    y = np.array(labels_list)
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Label distribution: Real={np.sum(y)}, Fake={len(y)-np.sum(y)}")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Normalize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    print("\n" + "="*80)
    print("TDA-ONLY MODEL EVALUATION")
    print("="*80)
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    }
    
    for name, model in models.items():
        print(f"\n{name}:")
        print("-" * 40)
        
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print(f"F1 Score: {f1:.4f}")
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Fake', 'Real']))
        print(f"\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        # Feature importance for RF
        if name == 'Random Forest':
            print(f"\nTop 10 Most Important Features:")
            feature_names = [
                'b0_integral', 'b0_max', 'b0_mean', 'b0_std', 'b0_peak_time', 'b0_duration',
                'b1_integral', 'b1_max', 'b1_mean', 'b1_std', 'b1_peak_time', 'b1_duration',
                'landscape_b0_l1', 'landscape_b0_l2', 'landscape_b0_l3', 'landscape_b0_l4',
                'landscape_b1_l1', 'landscape_b1_l2', 'landscape_b1_l3', 'landscape_b1_l4',
                'fiedler_value', 'algebraic_connectivity', 'spectral_gap', 'num_components',
                'avg_clustering', 'transitivity', 'assortativity', 'diameter',
                'growth_rate', 'burst_intensity', 'cascade_depth', 'cascade_breadth',
                'time_to_peak', 'decay_rate', 'structural_virality', 'wiener_index'
            ]
            
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1][:10]
            
            for i, idx in enumerate(indices):
                print(f"  {i+1}. {feature_names[idx]:25s}: {importances[idx]:.4f}")
    
    # Test SC-Logic rules
    print("\n" + "="*80)
    print("SC-LOGIC RULE-BASED CLASSIFICATION")
    print("="*80)
    
    rule_engine = SCLogicRuleEngine()
    
    predictions = []
    confidences = []
    
    for features in X:
        # Convert features to dict (simplified - using key features only)
        feature_dict = {
            'growth_rate': features[28],
            'max_beta1': features[7],
            'burst_intensity': features[29]
        }
        
        is_fake, confidence, explanation = rule_engine.apply_fake_rule(feature_dict)
        predictions.append(0 if is_fake else 1)
        confidences.append(confidence)
    
    f1_rules = f1_score(y, predictions, average='weighted')
    
    print(f"\nSC-Logic F1 Score: {f1_rules:.4f}")
    print(f"Mean Confidence: {np.mean(confidences):.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y, predictions, target_names=['Fake', 'Real']))
    print(f"\nConfusion Matrix:")
    print(confusion_matrix(y, predictions))
    
    # Final summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nDataset: 200 events (100 real, 100 fake)")
    print(f"Features: 36D advanced topological features")
    print(f"\nResults:")
    print(f"  Logistic Regression (TDA-only): F1 = {f1:.4f}")
    print(f"  Random Forest (TDA-only): F1 = {f1:.4f}")
    print(f"  SC-Logic Rules: F1 = {f1_rules:.4f}")
    
    print("\n" + "="*80)

if __name__ == '__main__':
    main()
