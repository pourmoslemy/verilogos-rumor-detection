"""
Shape Encoder Module

Encodes topological shapes into feature vectors for machine learning.
Maps truth shapes to numerical representations suitable for classification.

Author: Alireza Pourmoslemi
Email: apmath99@gmail.com
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from verilogos.applications.lie_detector.core.truth_geometry import (
    TruthShape,
    TopologicalSignature,
)


@dataclass
class ShapeFeatures:
    """
    Encoded shape features for ML.
    
    Attributes:
        topological: Topological invariants (Betti numbers, Euler char)
        persistence: Persistence-based features
        geometric: Geometric properties (fragmentation, coherence)
        shape_code_embedding: Encoded shape code
        feature_vector: Complete feature vector
    """
    topological: np.ndarray
    persistence: np.ndarray
    geometric: np.ndarray
    shape_code_embedding: np.ndarray
    feature_vector: np.ndarray
    
    def to_dict(self) -> Dict[str, np.ndarray]:
        """Convert to dictionary."""
        return {
            'topological': self.topological,
            'persistence': self.persistence,
            'geometric': self.geometric,
            'shape_code_embedding': self.shape_code_embedding,
            'feature_vector': self.feature_vector,
        }


class ShapeEncoder:
    """
    Encodes truth shapes into feature vectors.
    
    This class transforms topological shapes into numerical representations
    suitable for machine learning classifiers.
    
    Features extracted:
        - Betti numbers (dimensions 0, 1, 2)
        - Euler characteristic
        - Persistence statistics (entropy, total, max, mean, std)
        - Geometric properties (fragmentation, coherence, complexity, stability)
        - Shape code embedding (one-hot or learned)
    
    Example:
        >>> encoder = ShapeEncoder()
        >>> features = encoder.encode(truth_shape)
        >>> X = features.feature_vector
    """
    
    def __init__(self, max_dimension: int = 2):
        """
        Initialize shape encoder.
        
        Args:
            max_dimension: Maximum homology dimension
        """
        self.max_dimension = max_dimension
        self.shape_code_vocab = {}
        self.vocab_size = 0
        
    def encode(self, truth_shape: TruthShape) -> ShapeFeatures:
        """
        Encode truth shape into feature vector.
        
        Args:
            truth_shape: TruthShape object to encode
            
        Returns:
            ShapeFeatures with all encoded representations
        """
        # Extract topological features
        topo_features = self._encode_topological(truth_shape.signature)
        
        # Extract persistence features
        pers_features = self._encode_persistence(truth_shape.signature)
        
        # Extract geometric features
        geom_features = self._encode_geometric(truth_shape)
        
        # Encode shape code
        shape_embedding = self._encode_shape_code(truth_shape.shape_code)
        
        # Concatenate all features
        feature_vector = np.concatenate([
            topo_features,
            pers_features,
            geom_features,
            shape_embedding,
        ])
        
        return ShapeFeatures(
            topological=topo_features,
            persistence=pers_features,
            geometric=geom_features,
            shape_code_embedding=shape_embedding,
            feature_vector=feature_vector,
        )
    
    def encode_batch(self, truth_shapes: List[TruthShape]) -> np.ndarray:
        """
        Encode batch of truth shapes.
        
        Args:
            truth_shapes: List of TruthShape objects
            
        Returns:
            Feature matrix of shape (n_samples, n_features)
        """
        features = [self.encode(shape).feature_vector for shape in truth_shapes]
        return np.array(features)
    
    def _encode_topological(self, signature: TopologicalSignature) -> np.ndarray:
        """
        Encode topological invariants.
        
        Features:
            - Betti numbers for dimensions 0, 1, 2
            - Euler characteristic
            - Number of features per dimension
        """
        features = []
        
        # Betti numbers
        for dim in range(self.max_dimension + 1):
            features.append(signature.betti_numbers.get(dim, 0))
        
        # Euler characteristic
        features.append(signature.euler_characteristic)
        
        # Number of features per dimension
        for dim in range(self.max_dimension + 1):
            features.append(signature.num_features.get(dim, 0))
        
        return np.array(features, dtype=np.float32)
    
    def _encode_persistence(self, signature: TopologicalSignature) -> np.ndarray:
        """
        Encode persistence statistics.
        
        Features:
            - Persistence entropy
            - Total persistence
            - Max persistence
            - Mean persistence
            - Std persistence
            - Median persistence
        """
        features = [
            signature.persistence_entropy,
            signature.total_persistence,
            signature.max_persistence,
            signature.barcode_stats.get('mean', 0.0),
            signature.barcode_stats.get('std', 0.0),
            signature.barcode_stats.get('median', 0.0),
        ]
        
        return np.array(features, dtype=np.float32)
    
    def _encode_geometric(self, truth_shape: TruthShape) -> np.ndarray:
        """
        Encode geometric properties.
        
        Features:
            - Fragmentation score
            - Coherence score
            - Complexity score
            - Stability score
        """
        features = [
            truth_shape.fragmentation_score,
            truth_shape.coherence_score,
            truth_shape.complexity_score,
            truth_shape.stability_score,
        ]
        
        return np.array(features, dtype=np.float32)
    
    def _encode_shape_code(self, shape_code: str) -> np.ndarray:
        """
        Encode shape code into embedding.
        
        Uses simple hash-based embedding for now.
        Could be replaced with learned embeddings.
        
        Args:
            shape_code: Shape code string
            
        Returns:
            Embedding vector
        """
        # Add to vocabulary if new
        if shape_code not in self.shape_code_vocab:
            self.shape_code_vocab[shape_code] = self.vocab_size
            self.vocab_size += 1
        
        # Create one-hot-like embedding (simplified)
        # In production, use learned embeddings
        embedding_dim = 8
        embedding = np.zeros(embedding_dim, dtype=np.float32)
        
        # Hash shape code to embedding indices
        code_hash = hash(shape_code)
        for i in range(embedding_dim):
            embedding[i] = ((code_hash >> i) & 1) * 2 - 1
        
        return embedding
    
    def get_feature_names(self) -> List[str]:
        """
        Get names of all features.
        
        Returns:
            List of feature names
        """
        names = []
        
        # Topological features
        for dim in range(self.max_dimension + 1):
            names.append(f'betti_{dim}')
        names.append('euler_characteristic')
        for dim in range(self.max_dimension + 1):
            names.append(f'num_features_{dim}')
        
        # Persistence features
        names.extend([
            'persistence_entropy',
            'total_persistence',
            'max_persistence',
            'mean_persistence',
            'std_persistence',
            'median_persistence',
        ])
        
        # Geometric features
        names.extend([
            'fragmentation_score',
            'coherence_score',
            'complexity_score',
            'stability_score',
        ])
        
        # Shape code embedding
        for i in range(8):
            names.append(f'shape_embedding_{i}')
        
        return names
    
    def get_feature_importance_groups(self) -> Dict[str, List[int]]:
        """
        Get feature indices grouped by type.
        
        Useful for feature importance analysis.
        
        Returns:
            Dictionary mapping group name to feature indices
        """
        groups = {}
        idx = 0
        
        # Topological group
        topo_size = (self.max_dimension + 1) * 2 + 1
        groups['topological'] = list(range(idx, idx + topo_size))
        idx += topo_size
        
        # Persistence group
        pers_size = 6
        groups['persistence'] = list(range(idx, idx + pers_size))
        idx += pers_size
        
        # Geometric group
        geom_size = 4
        groups['geometric'] = list(range(idx, idx + geom_size))
        idx += geom_size
        
        # Shape code group
        shape_size = 8
        groups['shape_code'] = list(range(idx, idx + shape_size))
        
        return groups
