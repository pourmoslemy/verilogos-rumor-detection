#!/usr/bin/env python3
"""
Complete Rigorous TDA Experiment Runner for ACL2017 Rumor Detection

This script:
1. Loads all 1154 events from Twitter15/16
2. Extracts 36D advanced topological features using rigorous pipeline
3. Trains models with cross-event validation
4. Evaluates TDA-only performance
5. Tests SC-Logic rules
6. Generates comprehensive analysis report
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import warnings
warnings.filterwarnings('ignore')

# Import our rigorous TDA pipeline
from rigorous_tda_pipeline import (
    RigorousTDAPipeline,
    SCLogicRuleEngine
)


def load_acl2017_dataset(base_path: str):
    """Load complete ACL2017 dataset with all events."""
    print("Loading ACL2017 dataset...")
    
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
                # Map labels: true/false are the actual rumors, non-rumor is real news
                labels[tweet_id] = 1 if label == 'true' else 0
        
        # Load trees
        tree_dir = dataset_path / 'tree'
        for tree_file in tree_dir.glob('*.txt'):
            tweet_id = tree_file.stem
            
            if tweet_id not in labels:
                continue
            
            # Parse tree structure
            edges = []
            with open(tree_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or '->' not in line:
                        continue
                    
                    try:
                        parent_part, child_part = line.split('->')
                        parent_data = parent_part.strip("[]'").split(',')
                        child_data = child_part.strip("[]'").split(',')
                        
                        if len(parent_data) >= 3 and len(child_data) >= 3:
                            parent_user = parent_data[0].strip().strip("' ")
                            parent_tweet = parent_data[1].strip().strip("' ")
                            parent_time = float(parent_data[2].strip().strip("' "))
                            
                            child_user = child_data[0].strip().strip("' ")
                            child_tweet = child_data[1].strip().strip("' ")
                            child_time = float(child_data[2].strip().strip("' "))
                            
                            edges.append({
                                'parent_user': parent_user,
                                'parent_tweet': parent_tweet,
                                'parent_time': parent_time,
                                'child_user': child_user,
                                'child_tweet': child_tweet,
                                'child_time': child_time
                            })
                    except Exception as e:
                        continue
            
            if edges:
                all_data.append({
                    'event_id': f"{dataset_name}_{tweet_id}",
                    'dataset': dataset_name,
                    'tweet_id': tweet_id,
                    'label': labels[tweet_id],
                    'tree_file': tree_file,
                    'edges': edges,
                    'num_edges': len(edges)
                })
    
    print(f"Loaded {len(all_data)} events")
    print(f"  Real news: {sum(1 for d in all_data if d['label'] == 1)}")
    print(f"  Fake news: {sum(1 for d in all_data if d['label'] == 0)}")
    
    return all_data


def extract_rigorous_tda_features(events_data):
    """Extract 36D advanced topological features for all events."""
    print("\nExtracting rigorous TDA features...")
    
    features_list = []
    labels_list = []
    event_ids = []
    
    # Initialize pipeline with time-decay parameter
    pipeline = RigorousTDAPipeline(lambda_decay=0.001, temporal_window=60.0)
    
    for i, event in enumerate(events_data):
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(events_data)} events...")
        
        try:
            # Process cascade using the rigorous pipeline
            features, metadata = pipeline.process_cascade(event['tree_file'])
            
            features_list.append(features)
            labels_list.append(event['label'])
            event_ids.append(event['event_id'])
            
        except Exception as e:
            print(f"  Warning: Failed to process event {event['event_id']}: {e}")
            continue
    
    print(f"Successfully extracted features for {len(features_list)} events")
    
    # Convert to DataFrame
    feature_names = [
        # Betti curve features (12 features)
        'b0_integral', 'b0_max', 'b0_mean', 'b0_std', 'b0_peak_time', 'b0_duration',
        'b1_integral', 'b1_max', 'b1_mean', 'b1_std', 'b1_peak_time', 'b1_duration',
        
        # Persistence landscape features (8 features)
        'landscape_b0_l1', 'landscape_b0_l2', 'landscape_b0_l3', 'landscape_b0_l4',
        'landscape_b1_l1', 'landscape_b1_l2', 'landscape_b1_l3', 'landscape_b1_l4',
        
        # Graph topology features (8 features)
        'fiedler_value', 'algebraic_connectivity', 'spectral_gap', 'num_components',
        'avg_clustering', 'transitivity', 'assortativity', 'diameter',
        
        # Temporal dynamics (8 features)
        'growth_rate', 'burst_intensity', 'cascade_depth', 'cascade_breadth',
        'time_to_peak', 'decay_rate', 'structural_virality', 'wiener_index'
    ]
    
    df = pd.DataFrame(features_list, columns=feature_names)
    df['label'] = labels_list
    df['event_id'] = event_ids
    
    return df


def evaluate_tda_only(df):
    """Evaluate TDA-only performance with cross-event validation."""
    print("\n" + "="*80)
    print("EVALUATING TDA-ONLY PERFORMANCE")
    print("="*80)
    
    # Prepare data
    X = df.drop(['label', 'event_id'], axis=1).values
    y = df['label'].values
    groups = df['event_id'].values
    
    # Normalize features
    scaler = StandardScaler()
    
    # Cross-event validation (5-fold)
    gkf = GroupKFold(n_splits=5)
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    }
    
    results = {}
    
    for model_name, model in models.items():
        print(f"\n{model_name}:")
        print("-" * 40)
        
        fold_scores = []
        all_preds = []
        all_true = []
        
        for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Normalize
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train
            model.fit(X_train_scaled, y_train)
            
            # Predict
            y_pred = model.predict(X_test_scaled)
            
            # Score
            f1 = f1_score(y_test, y_pred, average='weighted')
            fold_scores.append(f1)
            
            all_preds.extend(y_pred)
            all_true.extend(y_test)
            
            print(f"  Fold {fold + 1}: F1 = {f1:.4f}")
        
        # Overall metrics
        print(f"\n  Mean F1: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")
        print(f"\n  Classification Report:")
        print(classification_report(all_true, all_preds, target_names=['Fake', 'Real']))
        
        print(f"\n  Confusion Matrix:")
        cm = confusion_matrix(all_true, all_preds)
        print(cm)
        
        results[model_name] = {
            'mean_f1': np.mean(fold_scores),
            'std_f1': np.std(fold_scores),
            'fold_scores': fold_scores
        }
        
        # Feature importance for Random Forest
        if model_name == 'Random Forest':
            print(f"\n  Top 10 Most Important Features:")
            feature_names = df.drop(['label', 'event_id'], axis=1).columns
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1][:10]
            
            for i, idx in enumerate(indices):
                print(f"    {i+1}. {feature_names[idx]}: {importances[idx]:.4f}")
    
    return results


def test_sc_logic_rules(df):
    """Test SC-Logic rule-based classification."""
    print("\n" + "="*80)
    print("TESTING SC-LOGIC RULES")
    print("="*80)
    
    rule_engine = SCLogicRuleEngine()
    
    correct = 0
    total = len(df)
    
    predictions = []
    confidences = []
    
    for _, row in df.iterrows():
        features = row.drop(['label', 'event_id']).to_dict()
        prediction, confidence, explanation = rule_engine.classify(features)
        
        predictions.append(prediction)
        confidences.append(confidence)
        
        if prediction == row['label']:
            correct += 1
    
    accuracy = correct / total
    f1 = f1_score(df['label'].values, predictions, average='weighted')
    
    print(f"\nSC-Logic Rule-Based Classification:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  Mean Confidence: {np.mean(confidences):.4f}")
    
    print(f"\nClassification Report:")
    print(classification_report(df['label'].values, predictions, target_names=['Fake', 'Real']))
    
    print(f"\nConfusion Matrix:")
    cm = confusion_matrix(df['label'].values, predictions)
    print(cm)
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'mean_confidence': np.mean(confidences)
    }


def analyze_topology_truth_correlation(df):
    """Analyze correlation between topological features and truth/falsehood."""
    print("\n" + "="*80)
    print("TOPOLOGY-TRUTH CORRELATION ANALYSIS")
    print("="*80)
    
    feature_cols = df.drop(['label', 'event_id'], axis=1).columns
    
    print("\nFeature Statistics by Label:")
    print("-" * 80)
    
    for label_val, label_name in [(0, 'FAKE'), (1, 'REAL')]:
        subset = df[df['label'] == label_val]
        print(f"\n{label_name} News (n={len(subset)}):")
        
        key_features = ['b1_integral', 'b1_max', 'fiedler_value', 'growth_rate', 
                       'burst_intensity', 'structural_virality']
        
        for feat in key_features:
            if feat in subset.columns:
                mean_val = subset[feat].mean()
                std_val = subset[feat].std()
                print(f"  {feat:25s}: {mean_val:8.4f} ± {std_val:6.4f}")
    
    # Compute correlations
    print("\n\nTop 10 Features Correlated with Truth (label=1):")
    print("-" * 80)
    
    correlations = []
    for feat in feature_cols:
        corr = df[feat].corr(df['label'])
        correlations.append((feat, corr))
    
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    
    for i, (feat, corr) in enumerate(correlations[:10]):
        direction = "→ REAL" if corr > 0 else "→ FAKE"
        print(f"  {i+1:2d}. {feat:25s}: {corr:+7.4f}  {direction}")


def main():
    """Main experiment runner."""
    print("="*80)
    print("RIGOROUS TDA EXPERIMENT FOR ACL2017 RUMOR DETECTION")
    print("="*80)
    
    # Configuration
    data_path = '/mnt/d/Verilogos/historical_data/rumor_detection_acl2017'
    
    # Step 1: Load dataset
    events_data = load_acl2017_dataset(data_path)
    
    # Step 2: Extract rigorous TDA features
    df = extract_rigorous_tda_features(events_data)
    
    # Save features
    output_file = 'rigorous_tda_features.csv'
    df.to_csv(output_file, index=False)
    print(f"\nSaved features to {output_file}")
    
    # Step 3: Evaluate TDA-only performance
    tda_results = evaluate_tda_only(df)
    
    # Step 4: Test SC-Logic rules
    sc_logic_results = test_sc_logic_rules(df)
    
    # Step 5: Analyze topology-truth correlation
    analyze_topology_truth_correlation(df)
    
    # Final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    
    print("\nTDA-Only Performance:")
    for model_name, results in tda_results.items():
        print(f"  {model_name}: F1 = {results['mean_f1']:.4f} ± {results['std_f1']:.4f}")
    
    print(f"\nSC-Logic Rules Performance:")
    print(f"  Accuracy: {sc_logic_results['accuracy']:.4f}")
    print(f"  F1 Score: {sc_logic_results['f1']:.4f}")
    
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
