"""
Topology Features - Extract topological features from text

Computes Betti numbers, persistence statistics, and topological signatures.

Author: Alireza Pourmoslemi
Email: apmath99@gmail.com
"""

import numpy as np
from typing import Dict, List, Any
from collections import defaultdict


def extract_topology_features(
    text: str,
    similarity_threshold: float = 0.3,
    max_dimension: int = 2
) -> Dict[str, float]:
    """
    Extract topological features from text.
    
    Uses VeriLogos core topology modules to compute:
    - Betti numbers (B0, B1, B2)
    - Persistence statistics
    - Topological signatures
    
    Args:
        text: Input text
        similarity_threshold: Threshold for edge creation
        max_dimension: Maximum homology dimension
    
    Returns:
        Dictionary of feature name -> value
    """
    features = {}
    
    try:
        # Import VeriLogos topology modules
        from verilogos.applications.lie_detector.topology.text_complex_builder import TextComplexBuilder
        from verilogos.applications.lie_detector.features.persistence_features import PersistenceFeatureExtractor
        
        # Build simplicial complex
        builder = TextComplexBuilder(
            similarity_threshold=similarity_threshold,
            max_dimension=max_dimension
        )
        complex = builder.build_complex(text)
        
        # Extract persistence features
        extractor = PersistenceFeatureExtractor(max_dimension=max_dimension)
        persistence_features = extractor.extract(complex)
        
        # Add all persistence features
        features.update(persistence_features)
        
    except Exception as e:
        print(f"Warning: Could not extract topology features: {e}")
        # Return default features
        features = _get_default_features(max_dimension)
    
    return features


def _get_default_features(max_dimension: int = 2) -> Dict[str, float]:
    """
    Get default feature values when extraction fails.
    
    Args:
        max_dimension: Maximum dimension
    
    Returns:
        Dictionary of default features
    """
    features = {}
    
    # Betti numbers
    for dim in range(max_dimension + 1):
        features[f'betti_{dim}'] = 0.0
    
    # Persistence statistics
    features['persistence_entropy'] = 0.0
    features['max_persistence'] = 0.0
    features['mean_persistence'] = 0.0
    features['std_persistence'] = 0.0
    features['total_persistence'] = 0.0
    
    # Topological properties
    features['euler_characteristic'] = 0.0
    features['num_simplices'] = 0.0
    features['num_vertices'] = 0.0
    features['num_edges'] = 0.0
    
    return features


def extract_batch_topology_features(
    texts: List[str],
    similarity_threshold: float = 0.3,
    max_dimension: int = 2,
    show_progress: bool = True
) -> np.ndarray:
    """
    Extract topology features for multiple texts.
    
    Args:
        texts: List of texts
        similarity_threshold: Threshold for edge creation
        max_dimension: Maximum dimension
        show_progress: Show progress bar
    
    Returns:
        Feature matrix (n_samples, n_features)
    """
    all_features = []
    
    for i, text in enumerate(texts):
        if show_progress and i % 10 == 0:
            print(f"Extracting features: {i}/{len(texts)}")
        
        features = extract_topology_features(
            text,
            similarity_threshold=similarity_threshold,
            max_dimension=max_dimension
        )
        
        all_features.append(features)
    
    # Convert to matrix
    if all_features:
        # Get feature names from first sample
        feature_names = sorted(all_features[0].keys())
        
        # Create matrix
        X = np.zeros((len(all_features), len(feature_names)))
        for i, features in enumerate(all_features):
            X[i] = [features.get(name, 0.0) for name in feature_names]
        
        return X, feature_names
    else:
        return np.array([]), []


def compute_betti_numbers(complex) -> Dict[int, int]:
    """
    Compute Betti numbers from simplicial complex.
    
    Args:
        complex: Simplicial complex
    
    Returns:
        Dictionary mapping dimension -> Betti number
    """
    try:
        from verilogos.core.topology import PersistentHomology
        
        ph = PersistentHomology(complex)
        barcodes = ph.compute_persistence()
        
        betti = defaultdict(int)
        for dim, intervals in barcodes.items():
            # Count infinite intervals (permanent features)
            betti[dim] = sum(1 for birth, death in intervals if death == float('inf'))
        
        return dict(betti)
    
    except Exception as e:
        print(f"Warning: Could not compute Betti numbers: {e}")
        return {0: 1, 1: 0, 2: 0}


def compute_persistence_statistics(barcode: List[tuple]) -> Dict[str, float]:
    """
    Compute statistics from persistence barcode.
    
    Args:
        barcode: List of (birth, death) tuples
    
    Returns:
        Dictionary of statistics
    """
    if not barcode:
        return {
            'max_persistence': 0.0,
            'mean_persistence': 0.0,
            'std_persistence': 0.0,
            'total_persistence': 0.0,
            'num_features': 0
        }
    
    # Filter out infinite intervals
    finite_intervals = [(b, d) for b, d in barcode if d != float('inf')]
    
    if not finite_intervals:
        return {
            'max_persistence': 0.0,
            'mean_persistence': 0.0,
            'std_persistence': 0.0,
            'total_persistence': 0.0,
            'num_features': len(barcode)
        }
    
    # Compute lifespans
    lifespans = [d - b for b, d in finite_intervals]
    
    return {
        'max_persistence': float(np.max(lifespans)),
        'mean_persistence': float(np.mean(lifespans)),
        'std_persistence': float(np.std(lifespans)),
        'total_persistence': float(np.sum(lifespans)),
        'num_features': len(barcode)
    }


def compute_persistence_entropy(barcode: List[tuple]) -> float:
    """
    Compute persistence entropy.
    
    Args:
        barcode: List of (birth, death) tuples
    
    Returns:
        Entropy value
    """
    if not barcode:
        return 0.0
    
    # Filter finite intervals
    finite_intervals = [(b, d) for b, d in barcode if d != float('inf')]
    
    if not finite_intervals:
        return 0.0
    
    # Compute lifespans
    lifespans = np.array([d - b for b, d in finite_intervals])
    
    # Normalize to probabilities
    total = lifespans.sum()
    if total == 0:
        return 0.0
    
    probs = lifespans / total
    
    # Compute entropy
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    
    return float(entropy)
