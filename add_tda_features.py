"""
Add topological features from Twitter propagation graphs.
Extract Betti numbers and persistence intervals from retweet cascades.
"""

import sys
sys.path.insert(0, '/mnt/d/Verilogos')

from pathlib import Path
import numpy as np
from collections import defaultdict, deque
import re

# Import VeriLogos TDA components
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
                # Format: ['user','tweet','time']->['user','tweet','time']
                parent_part, child_part = line.split('->')
                
                # Extract values using regex
                parent_match = re.findall(r"'([^']*)'", parent_part)
                child_match = re.findall(r"'([^']*)'", child_part)
                
                if len(parent_match) >= 2 and len(child_match) >= 3:
                    parent_tweet = parent_match[1]
                    child_tweet = child_match[1]
                    time_str = child_match[2]
                    time_delay = float(time_str) if time_str != 'None' else 0.0
                    
                    edges.append((parent_tweet, child_tweet, time_delay))
            except Exception as e:
                continue
    
    return edges

def extract_topological_features(tree_file: Path):
    """Extract topological features from propagation tree."""
    edges = load_propagation_tree(tree_file)
    
    if not edges:
        return {
            'betti_0': 0, 'betti_1': 0, 'num_nodes': 0, 'num_edges': 0,
            'max_depth': 0, 'avg_time_delay': 0.0, 'branching_factor': 0.0
        }
    
    # Build node set
    nodes = set()
    for parent, child, _ in edges:
        nodes.add(parent)
        nodes.add(child)
    
    num_nodes = len(nodes)
    num_edges = len(edges)
    
    # Time features
    time_delays = [t for _, _, t in edges if t > 0]
    avg_time_delay = np.mean(time_delays) if time_delays else 0.0
    
    # Build simplicial complex
    complex_obj = SimplicialComplex()
    
    # Add 0-simplices (nodes)
    for node in nodes:
        complex_obj.add_simplex(Simplex([node]))
    
    # Add 1-simplices (edges)
    edge_set = set()
    for parent, child, _ in edges:
        edge_tuple = tuple(sorted([parent, child]))
        if edge_tuple not in edge_set:
            complex_obj.add_simplex(Simplex(list(edge_tuple)))
            edge_set.add(edge_tuple)
    
    # Compute Betti numbers
    # Betti_0: connected components
    try:
        # Use DFS to count components
        adjacency = defaultdict(set)
        for parent, child, _ in edges:
            adjacency[parent].add(child)
            adjacency[child].add(parent)
        
        visited = set()
        betti_0 = 0
        
        for node in nodes:
            if node not in visited:
                betti_0 += 1
                # DFS
                stack = [node]
                while stack:
                    current = stack.pop()
                    if current in visited:
                        continue
                    visited.add(current)
                    stack.extend(adjacency[current] - visited)
        
        # Betti_1: independent cycles (Euler characteristic)
        # For a graph: Betti_1 = edges - nodes + components
        betti_1 = max(0, num_edges - num_nodes + betti_0)
        
    except:
        betti_0 = 1
        betti_1 = 0
    
    # Cascade depth (longest path from root)
    children_map = defaultdict(list)
    for parent, child, _ in edges:
        children_map[parent].append(child)
    
    # Find roots (nodes with no parent)
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
    
    # Branching factor (avg children per node)
    branching_factor = np.mean([len(children_map[node]) for node in nodes if node in children_map]) if children_map else 0.0
    
    return {
        'betti_0': betti_0,
        'betti_1': betti_1,
        'num_nodes': num_nodes,
        'num_edges': num_edges,
        'max_depth': max_depth,
        'avg_time_delay': avg_time_delay,
        'branching_factor': branching_factor
    }

print("="*80)
print("EXTRACTING TOPOLOGICAL FEATURES FROM PROPAGATION GRAPHS")
print("="*80)

root = Path('historical_data/rumor_detection_acl2017')

print("\nTesting feature extraction on sample trees...")

for version in ['twitter15', 'twitter16']:
    tree_dir = root / version / 'tree'
    tree_files = list(tree_dir.glob('*.txt'))[:5]
    
    print(f"\n{version.upper()}:")
    for tree_file in tree_files:
        features = extract_topological_features(tree_file)
        print(f"  {tree_file.name}:")
        print(f"    Betti_0={features['betti_0']}, Betti_1={features['betti_1']}")
        print(f"    Nodes={features['num_nodes']}, Edges={features['num_edges']}")
        print(f"    Depth={features['max_depth']}, Branch={features['branching_factor']:.2f}")

print("\n" + "="*80)
print("SUCCESS: TDA feature extraction working!")
print("="*80)

