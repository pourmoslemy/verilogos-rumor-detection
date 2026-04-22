"""
Full experiment with advanced TDA features.
"""

import sys
sys.path.insert(0, '/mnt/d/Verilogos')

from pathlib import Path
import numpy as np
from collections import defaultdict
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# Import advanced TDA functions
exec(open('advanced_tda_features.py').read().split('print("="*80)')[0])

def load_acl2017_with_advanced_tda(root: Path, version: str):
    """Load ACL2017 data with advanced TDA features."""
    version_dir = root / version
    label_file = version_dir / "label.txt"
    source_file = version_dir / "source_tweets.txt"
    tree_dir = version_dir / "tree"
    
    source_texts = {}
    with source_file.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or "\t" not in line:
                continue
            tweet_id, text = line.split("\t", 1)
            source_texts[tweet_id.strip()] = text.strip()
    
    data = []
    with label_file.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            label_str, tweet_id = [x.strip() for x in line.split(":", 1)]
            
            if label_str.lower() == "true":
                label = 0
            elif label_str.lower() == "false":
                label = 1
            else:
                continue
            
            text = source_texts.get(tweet_id, "")
            if not text:
                continue
            
            # Extract advanced TDA features
            tree_file = tree_dir / f"{tweet_id}.txt"
            if tree_file.exists():
                tda_features = extract_advanced_tda_features(tree_file)
            else:
                tda_features = np.zeros(10)
            
            data.append({
                'event_id': tweet_id,
                'text': text,
                'label': label,
                'tda_features': tda_features
            })
    
    return data

print("="*80)
print("ADVANCED TDA EXPERIMENT: USER INTERACTION GRAPHS + TEMPORAL FILTRATION")
print("="*80)

# Load data
root = Path('historical_data/rumor_detection_acl2017')
print("\nLoading data with advanced TDA features...")
data_t15 = load_acl2017_with_advanced_tda(root, 'twitter15')
data_t16 = load_acl2017_with_advanced_tda(root, 'twitter16')
all_data = data_t15 + data_t16

print(f"Loaded {len(all_data)} samples")

# Cross-event split
fake_events = [d for d in all_data if d['label'] == 1]
real_events = [d for d in all_data if d['label'] == 0]

np.random.seed(42)
np.random.shuffle(fake_events)
np.random.shuffle(real_events)

n_fake_train = int(0.8 * len(fake_events))
n_real_train = int(0.8 * len(real_events))

train_data = fake_events[:n_fake_train] + real_events[:n_real_train]
test_data = fake_events[n_fake_train:] + real_events[n_real_train:]

np.random.shuffle(train_data)
np.random.shuffle(test_data)

print(f"Train: {len(train_data)}, Test: {len(test_data)}")

# Extract features
X_train_text = [d['text'] for d in train_data]
X_test_text = [d['text'] for d in test_data]
y_train = np.array([d['label'] for d in train_data])
y_test = np.array([d['label'] for d in test_data])

X_train_tda = np.array([d['tda_features'] for d in train_data])
X_test_tda = np.array([d['tda_features'] for d in test_data])

# Normalize TDA features
scaler = StandardScaler()
X_train_tda_scaled = scaler.fit_transform(X_train_tda)
X_test_tda_scaled = scaler.transform(X_test_tda)

# Text features
vectorizer = TfidfVectorizer(max_features=2000, stop_words='english', min_df=2)
X_train_text_vec = vectorizer.fit_transform(X_train_text).toarray()
X_test_text_vec = vectorizer.transform(X_test_text).toarray()

# Combine
X_train_combined = np.hstack([X_train_text_vec, X_train_tda_scaled])
X_test_combined = np.hstack([X_test_text_vec, X_test_tda_scaled])

print(f"\nFeature dimensions:")
print(f"  Text: {X_train_text_vec.shape[1]}")
print(f"  Advanced TDA: {X_train_tda_scaled.shape[1]}")
print(f"  Combined: {X_train_combined.shape[1]}")

print(f"\n{'='*80}")
print("RESULTS")
print("="*80)

# 1. Text-only baseline
lr_text = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
lr_text.fit(X_train_text_vec, y_train)
y_pred_text = lr_text.predict(X_test_text_vec)

print(f"\n1. TEXT-ONLY (Baseline)")
print(f"   F1: {f1_score(y_test, y_pred_text, average='binary'):.4f}")
print(f"   Accuracy: {accuracy_score(y_test, y_pred_text):.4f}")

# 2. Advanced TDA-only
lr_tda = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
lr_tda.fit(X_train_tda_scaled, y_train)
y_pred_tda = lr_tda.predict(X_test_tda_scaled)

print(f"\n2. ADVANCED TDA-ONLY (User Graphs + Temporal Filtration)")
print(f"   F1: {f1_score(y_test, y_pred_tda, average='binary'):.4f}")
print(f"   Accuracy: {accuracy_score(y_test, y_pred_tda):.4f}")
print(classification_report(y_test, y_pred_tda, target_names=['Real', 'Fake']))

# 3. Text + Advanced TDA
lr_combined = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
lr_combined.fit(X_train_combined, y_train)
y_pred_combined = lr_combined.predict(X_test_combined)

print(f"\n3. TEXT + ADVANCED TDA (Hybrid)")
print(f"   F1: {f1_score(y_test, y_pred_combined, average='binary'):.4f}")
print(f"   Accuracy: {accuracy_score(y_test, y_pred_combined):.4f}")
print(classification_report(y_test, y_pred_combined, target_names=['Real', 'Fake']))

# 4. Random Forest
rf_combined = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', max_depth=15)
rf_combined.fit(X_train_combined, y_train)
y_pred_rf = rf_combined.predict(X_test_combined)

print(f"\n4. RANDOM FOREST (Text + Advanced TDA)")
print(f"   F1: {f1_score(y_test, y_pred_rf, average='binary'):.4f}")
print(f"   Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")

# Feature importance
print(f"\n{'='*80}")
print("ADVANCED TDA FEATURE IMPORTANCE")
print("="*80)

tda_coefs = lr_combined.coef_[0][-10:]
tda_names = [
    'β₀_features', 'β₁_features', 'max_β₀_life', 'max_β₁_life',
    'avg_β₀_life', 'cycle_density', 'interactions', 'users', 'max_time', 'avg_time'
]

print("\nAdvanced TDA coefficients:")
for name, coef in zip(tda_names, tda_coefs):
    print(f"  {name:15s}: {coef:+.4f}")

# TDA-only feature importance
print("\nTDA-only model coefficients:")
for name, coef in zip(tda_names, lr_tda.coef_[0]):
    print(f"  {name:15s}: {coef:+.4f}")

print(f"\n{'='*80}")
print("CONCLUSION")
print("="*80)
print(f"Baseline (Text-only):        {f1_score(y_test, y_pred_text, average='binary'):.4f}")
print(f"Advanced TDA-only:           {f1_score(y_test, y_pred_tda, average='binary'):.4f}")
print(f"Hybrid (Text + Adv TDA):     {f1_score(y_test, y_pred_combined, average='binary'):.4f}")

improvement_tda = f1_score(y_test, y_pred_tda, average='binary') - 0.4792
improvement_hybrid = f1_score(y_test, y_pred_combined, average='binary') - f1_score(y_test, y_pred_text, average='binary')

print(f"\nImprovement over simple TDA: {improvement_tda:+.4f} ({improvement_tda*100:+.2f}%)")
print(f"Improvement over text-only:  {improvement_hybrid:+.4f} ({improvement_hybrid*100:+.2f}%)")

