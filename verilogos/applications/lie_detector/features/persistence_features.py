"""
Persistence Features Module

Extracts features from persistent homology computations.

Author: Alireza Pourmoslemi
Email: apmath99@gmail.com
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

from verilogos.core.topology import SimplicialComplex
from verilogos.applications.lie_detector.core.truth_geometry import TruthGeometry, TruthShape


class PersistenceFeatureExtractor:
    """
    Extracts features from persistence barcodes and diagrams.
    
    Features computed:
        - Betti numbers (b0, b1, b2)
        - Euler characteristic
        - Persistence statistics (entropy, total, max, mean, std)
        - Barcode lifespans
        - Birth-death histograms
        - Fragmentation index
        - Stability measures
    
    Example:
        >>> extractor = PersistenceFeatureExtractor()
        >>> features = extractor.extract(complex)
        >>> print(features.keys())
    """
    
    def __init__(self, max_dimension: int = 2):
        """
        Initialize feature extractor.
        
        Args:
            max_dimension: Maximum homology dimension
        """
        self.max_dimension = max_dimension
        self.truth_geometry = TruthGeometry(max_dimension=max_dimension)
    
    def extract(self, complex: SimplicialComplex) -> Dict[str, float]:
        """
        Extract all persistence features from complex.
        
        Args:
            complex: SimplicialComplex to analyze
            
        Returns:
            Dictionary of feature name to value
        """
        # Compute truth shape
        truth_shape = self.truth_geometry.analyze(complex)
        
        # Extract features
        features = {}
        
        # Topological features
        features.update(self._extract_topological_features(truth_shape))
        
        # Persistence features
        features.update(self._extract_persistence_features(truth_shape))
        
        # Geometric features
        features.update(self._extract_geometric_features(truth_shape))
        
        # Statistical features
        features.update(self._extract_statistical_features(truth_shape))
        
        return features
    
    def _extract_topological_features(self, truth_shape: TruthShape) -> Dict[str, float]:
        """Extract topological invariants."""
        features = {}
        
        # Betti numbers
        for dim in range(self.max_dimension + 1):
            features[f'betti_{dim}'] = float(truth_shape.signature.betti_numbers.get(dim, 0))
        
        # Euler characteristic
        features['euler_characteristic'] = float(truth_shape.signature.euler_characteristic)
        
        # Number of features per dimension
        for dim in range(self.max_dimension + 1):
            features[f'num_features_{dim}'] = float(truth_shape.signature.num_features.get(dim, 0))
        
        return features
    
    def _extract_persistence_features(self, truth_shape: TruthShape) -> Dict[str, float]:
        """Extract persistence-based features."""
        features = {}
        
        sig = truth_shape.signature
        
        # Persistence statistics
        features['persistence_entropy'] = sig.persistence_entropy
        features['total_persistence'] = sig.total_persistence
        features['max_persistence'] = sig.max_persistence
        features['mean_persistence'] = sig.barcode_stats.get('mean', 0.0)
        features['std_persistence'] = sig.barcode_stats.get('std', 0.0)
        features['median_persistence'] = sig.barcode_stats.get('median', 0.0)
        
        return features
    
    def _extract_geometric_features(self, truth_shape: TruthShape) -> Dict[str, float]:
        """Extract geometric properties."""
        return {
            'fragmentation_score': truth_shape.fragmentation_score,
            'coherence_score': truth_shape.coherence_score,
            'complexity_score': truth_shape.complexity_score,
            'stability_score': truth_shape.stability_score,
        }
    
    def _extract_statistical_features(self, truth_shape: TruthShape) -> Dict[str, float]:
        """Extract statistical features."""
        features = {}
        
        # Ratios
        b0 = truth_shape.signature.betti_numbers.get(0, 0)
        b1 = truth_shape.signature.betti_numbers.get(1, 0)
        b2 = truth_shape.signature.betti_numbers.get(2, 0)
        
        features['b1_b0_ratio'] = b1 / b0 if b0 > 0 else 0.0
        features['b2_b1_ratio'] = b2 / b1 if b1 > 0 else 0.0
        
        # Complexity measures
        total_features = b0 + b1 + b2
        features['total_features'] = float(total_features)
        features['feature_diversity'] = len([b for b in [b0, b1, b2] if b > 0]) / 3.0
        
        return features
    
    def extract_barcode_lifespans(
        self,
        barcodes: Dict[int, List[Tuple[float, Optional[float]]]]
    ) -> Dict[str, np.ndarray]:
        """
        Extract lifespan distributions from barcodes.
        
        Args:
            barcodes: Dictionary mapping dimension to list of (birth, death) tuples
            
        Returns:
            Dictionary mapping dimension to lifespan array
        """
        lifespans = {}
        
        for dim, intervals in barcodes.items():
            dim_lifespans = []
            for birth, death in intervals:
                if death is not None:
                    lifespan = death - birth
                    dim_lifespans.append(lifespan)
            
            lifespans[dim] = np.array(dim_lifespans) if dim_lifespans else np.array([])
        
        return lifespans
    
    def compute_birth_death_histogram(
        self,
        barcodes: Dict[int, List[Tuple[float, Optional[float]]]],
        bins: int = 10
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Compute histograms of birth and death times.
        
        Args:
            barcodes: Persistence barcodes
            bins: Number of histogram bins
            
        Returns:
            Dictionary with birth and death histograms per dimension
        """
        histograms = {}
        
        for dim, intervals in barcodes.items():
            births = [birth for birth, death in intervals]
            deaths = [death for birth, death in intervals if death is not None]
            
            if births:
                birth_hist, birth_edges = np.histogram(births, bins=bins)
                histograms[f'dim{dim}_birth'] = (birth_hist, birth_edges)
            
            if deaths:
                death_hist, death_edges = np.histogram(deaths, bins=bins)
                histograms[f'dim{dim}_death'] = (death_hist, death_edges)
        
        return histograms
    
    def compute_fragmentation_index(self, truth_shape: TruthShape) -> float:
        """
        Compute fragmentation index.
        
        High fragmentation indicates disconnected narrative structure.
        
        Args:
            truth_shape: TruthShape object
            
        Returns:
            Fragmentation index in [0, 1]
        """
        b0 = truth_shape.signature.betti_numbers.get(0, 1)
        total_features = sum(truth_shape.signature.num_features.values())
        
        if total_features == 0:
            return 0.0
        
        # Fragmentation is ratio of components to total features
        fragmentation = b0 / total_features
        
        return min(fragmentation, 1.0)
    
    def compute_persistence_landscape(
        self,
        barcodes: Dict[int, List[Tuple[float, Optional[float]]]],
        resolution: int = 100
    ) -> Dict[int, np.ndarray]:
        """
        Compute persistence landscape representation.
        
        Args:
            barcodes: Persistence barcodes
            resolution: Number of points in landscape
            
        Returns:
            Dictionary mapping dimension to landscape array
        """
        landscapes = {}
        
        for dim, intervals in barcodes.items():
            if not intervals:
                landscapes[dim] = np.zeros(resolution)
                continue
            
            # Find time range
            births = [b for b, d in intervals]
            deaths = [d for b, d in intervals if d is not None]
            
            if not deaths:
                landscapes[dim] = np.zeros(resolution)
                continue
            
            t_min = min(births)
            t_max = max(deaths)
            t_range = np.linspace(t_min, t_max, resolution)
            
            # Compute landscape function
            landscape = np.zeros(resolution)
            for birth, death in intervals:
                if death is None:
                    continue
                for i, t in enumerate(t_range):
                    if birth <= t <= death:
                        # Tent function
                        if t <= (birth + death) / 2:
                            landscape[i] += t - birth
                        else:
                            landscape[i] += death - t
            
            landscapes[dim] = landscape
        
        return landscapes
