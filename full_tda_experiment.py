"""
Full experiment: Text + TDA features on ACL2017 with cross-event validation.
"""

import sys
sys.path.insert(0, '/mnt/d/Verilogos')

from pathlib import Path
import numpy as np
from collections import defaultdict, deque
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, classification_report
from verilogos.core.topology.complexes.complex import SimplicialComplex, Simplex

def load_propagation_tree(tree_file: Path):
    """Load propagation tree from ACL2017 format."""
    edges = []
    with tree_file.open('r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if '->' not in line:
                continue
            try:
                parent_part, child_part = line.split('->')
                parent_match = re.findall(r"'([^']*)'", parent_part)
                child_match = re.findall(r"'([^']*)'", child_part)
                
                if len(parent_match) >= 2 and len(child_match) >= 3:
                    parent_tweet = parent_match[1]
                    child_tweet = child_match[1]
                    time_str = child_match[2]
                    time_delay = float(time_str) if time_str != 'None' else 0.0
                    edges.append((parent_tweet, child_tweet, time_delay))
            except:
                continue
    return edges

def extract_topological_features(tree_file: Path):
    """Extract topological features from propagation tree."""
    edges = load_propagation_tree(tree_file)
    
    if not edges:
        return np.zeros(7)
    
    nodes = set()
    for parent, child, _ in edges:
        nodes.add(parent)
        nodes.add(child)
    
    num_nodes = len(nodes)
    num_edges = len(edges)
    
    time_delays = [t for _, _, t in edges if t > 0]
    avg_time_delay = np.mean(time_delays) if time_delays else 0.0
    
    # Betti numbers
    adjacency = defaultdict(set)
    for parent, child, _ in edges:
        adjacency[parent].add(child)
        adjacency[child].add(parent)
    
    visited = set()
    betti_0 = 0
    for node in nodes:
        if node not in visited:
            betti_0 += 1
            stack = [node]
            while stack:
                current = stack.pop()
                if current in visited:
                    continue
                visited.add(current)
                stack.extend(adjacency[current] - visited)
    
    betti_1 = max(0, num_edges - num_nodes + betti_0)
    
    # Cascade depth
    children_map = defaultdict(list)
    for parent, child, _ in edges:
        children_map[parent].append(child)
    
    all_children = set(child for _, child, _ in edges)
    all_parents = set(parent for parent, _, _ in edges)
    roots = all_parents - all_children
    
    max_depth = 0
    if roots:
        for root in roots:
            queue = deque([(root, 0)])
            visited = set()
            while queue:
                node, depth = queue.popleft()
                if node in visited:
                    continue
                visited.add(node)
                max_depth = max(max_depth, depth)
                for child in children_map[node]:
                    queue.append((child, depth + 1))
    
    branching_factor = np.mean([len(children_map[node]) for node in nodes if node in children_map]) if children_map else 0.0
    
    return np.array([
        betti_0, betti_1, num_nodes, num_edges, 
        max_depth, avg_time_delay, branching_factor
    ])

def load_acl2017_with_tda(root: Path, version: str):
    """Load ACL2017 data with TDA features."""
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
            
            # Extract TDA features
            tree_file = tree_dir / f"{tweet_id}.txt"
            if tree_file.exists():
                tda_features = extract_topological_features(tree_file)
            else:
                tda_features = np.zeros(7)
            
            data.append({
                'event_id': tweet_id,
                'text': text,
                'label': label,
                'tda_features': tda_features
            })
    
    return data

print("="*80)
print("FULL TDA EXPERIMENT: TEXT + TOPOLOGY FEATURES")
print("="*80)

# Load data with TDA features
root = Path('historical_data/rumor_detection_acl2017')
print("\nLoading data with TDA features...")
data_t15 = load_acl2017_with_tda(root, 'twitter15')
data_t16 = load_acl2017_with_tda(root, 'twitter16')
all_data = data_t15 + data_t16

print(f"Loaded {len(all_data)} samples with TDA features")

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

# Text features
vectorizer = TfidfVectorizer(max_features=2000, stop_words='english', min_df=2)
X_train_text_vec = vectorizer.fit_transform(X_train_text).toarray()
X_test_text_vec = vectorizer.transform(X_test_text).toarray()

# Combine text + TDA
X_train_combined = np.hstack([X_train_text_vec, X_train_tda])
X_test_combined = np.hstack([X_test_text_vec, X_test_tda])

print(f"\nFeature dimensions:")
print(f"  Text: {X_train_text_vec.shape[1]}")
print(f"  TDA: {X_train_tda.shape[1]}")
print(f"  Combined: {X_train_combined.shape[1]}")

# Train models
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

# 2. TDA-only
lr_tda = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
lr_tda.fit(X_train_tda, y_train)
y_pred_tda = lr_tda.predict(X_test_tda)

print(f"\n2. TDA-ONLY (Topology)")
print(f"   F1: {f1_score(y_test, y_pred_tda, average='binary'):.4f}")
print(f"   Accuracy: {accuracy_score(y_test, y_pred_tda):.4f}")

# 3. Text + TDA combined
lr_combined = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
lr_combined.fit(X_train_combined, y_train)
y_pred_combined = lr_combined.predict(X_test_combined)

print(f"\n3. TEXT + TDA (Hybrid)")
print(f"   F1: {f1_score(y_test, y_pred_combined, average='binary'):.4f}")
print(f"   Accuracy: {accuracy_score(y_test, y_pred_combined):.4f}")
print(classification_report(y_test, y_pred_combined, target_names=['Real', 'Fake']))

# 4. Random Forest with combined features
rf_combined = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', max_depth=15)
rf_combined.fit(X_train_combined, y_train)
y_pred_rf = rf_combined.predict(X_test_combined)

print(f"\n4. RANDOM FOREST (Text + TDA)")
print(f"   F1: {f1_score(y_test, y_pred_rf, average='binary'):.4f}")
print(f"   Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")

# Feature importance analysis
print(f"\n{'='*80}")
print("TDA FEATURE IMPORTANCE (from combined LR model)")
print("="*80)

tda_coefs = lr_combined.coef_[0][-7:]
tda_names = ['Betti_0', 'Betti_1', 'Nodes', 'Edges', 'Depth', 'AvgTime', 'Branching']

print("\nTDA feature coefficients:")
for name, coef in zip(tda_names, tda_coefs):
    print(f"  {name:12s}: {coef:+.4f}")

print(f"\n{'='*80}")
print("CONCLUSION")
print("="*80)
print(f"Baseline (Text-only): {f1_score(y_test, y_pred_text, average='binary'):.4f}")
print(f"With TDA (Text+Topo): {f1_score(y_test, y_pred_combined, average='binary'):.4f}")
improvement = f1_score(y_test, y_pred_combined, average='binary') - f1_score(y_test, y_pred_text, average='binary')
print(f"Improvement: {improvement:+.4f} ({improvement*100:+.2f}%)")

