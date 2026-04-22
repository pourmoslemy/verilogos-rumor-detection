"""
Truth Geometry Module

Implements topological analysis of narrative structures to detect misinformation.
Uses persistent homology and simplicial complex theory to compute truth geometry.

Author: Alireza Pourmoslemi
Email: apmath99@gmail.com
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict

from verilogos.core.topology import (
    SimplicialComplex,
    Filtration,
    PersistentHomology,
    PersistenceInterval,
    Barcode,
)
from verilogos.core.reasoning import PersistenceEngine


@dataclass
class TopologicalSignature:
    """
    Topological signature of a narrative.
    
    Attributes:
        betti_numbers: Betti numbers for dimensions 0, 1, 2
        euler_characteristic: Euler characteristic
        persistence_entropy: Entropy of persistence diagram
        total_persistence: Sum of all persistence values
        max_persistence: Maximum persistence value
        num_features: Number of topological features per dimension
        barcode_stats: Statistics of persistence barcodes
    """
    betti_numbers: Dict[int, int] = field(default_factory=dict)
    euler_characteristic: int = 0
    persistence_entropy: float = 0.0
    total_persistence: float = 0.0
    max_persistence: float = 0.0
    num_features: Dict[int, int] = field(default_factory=dict)
    barcode_stats: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'betti_numbers': self.betti_numbers,
            'euler_characteristic': self.euler_characteristic,
            'persistence_entropy': self.persistence_entropy,
            'total_persistence': self.total_persistence,
            'max_persistence': self.max_persistence,
            'num_features': self.num_features,
            'barcode_stats': self.barcode_stats,
        }


@dataclass
class TruthShape:
    """
    Complete truth shape representation.
    
    Combines topological signature with geometric properties and
    narrative structure analysis.
    """
    signature: TopologicalSignature
    shape_code: str = ""
    fragmentation_score: float = 0.0
    coherence_score: float = 0.0
    complexity_score: float = 0.0
    stability_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'signature': self.signature.to_dict(),
            'shape_code': self.shape_code,
            'fragmentation_score': self.fragmentation_score,
            'coherence_score': self.coherence_score,
            'complexity_score': self.complexity_score,
            'stability_score': self.stability_score,
        }


class TruthGeometry:
    """
    Main class for computing truth geometry from simplicial complexes.
    
    This class implements the core topological analysis pipeline:
    1. Compute persistent homology
    2. Extract topological features
    3. Compute geometric properties
    4. Generate truth shape representation
    
    Example:
        >>> from verilogos.applications.lie_detector.core import TruthGeometry
        >>> geometry = TruthGeometry()
        >>> complex = build_complex_from_text(text)
        >>> truth_shape = geometry.analyze(complex)
        >>> print(truth_shape.shape_code)
    """
    
    def __init__(self, max_dimension: int = 2):
        """
        Initialize TruthGeometry analyzer.
        
        Args:
            max_dimension: Maximum homology dimension to compute
        """
        self.max_dimension = max_dimension
        self.persistence_engine = PersistenceEngine()
        
    def analyze(self, complex: SimplicialComplex) -> TruthShape:
        """
        Perform complete topological analysis of a simplicial complex.
        
        Args:
            complex: Simplicial complex representing narrative structure
            
        Returns:
            TruthShape object containing all topological properties
        """
        # Compute topological signature
        signature = self.compute_signature(complex)
        
        # Compute geometric properties
        fragmentation = self.compute_fragmentation(complex, signature)
        coherence = self.compute_coherence(complex, signature)
        complexity = self.compute_complexity(signature)
        stability = self.compute_stability(signature)
        
        # Generate shape code
        shape_code = self.generate_shape_code(signature, fragmentation, coherence)
        
        return TruthShape(
            signature=signature,
            shape_code=shape_code,
            fragmentation_score=fragmentation,
            coherence_score=coherence,
            complexity_score=complexity,
            stability_score=stability,
        )
    
    def compute_signature(self, complex: SimplicialComplex) -> TopologicalSignature:
        """
        Compute topological signature from simplicial complex.
        
        Args:
            complex: Input simplicial complex
            
        Returns:
            TopologicalSignature with all topological invariants
        """
        # Compute Betti numbers
        betti_numbers = self._compute_betti_numbers(complex)
        
        # Compute Euler characteristic
        euler_char = complex.euler_characteristic()
        
        # Create filtration for persistence computation
        filtration = self._create_filtration(complex)
        
        # Compute persistent homology
        barcodes = self.persistence_engine.compute_barcodes(filtration)
        
        # Extract persistence statistics
        persistence_stats = self._compute_persistence_stats(barcodes)
        
        # Count features per dimension
        num_features = {dim: len(intervals) for dim, intervals in barcodes.items()}
        
        return TopologicalSignature(
            betti_numbers=betti_numbers,
            euler_characteristic=euler_char,
            persistence_entropy=persistence_stats['entropy'],
            total_persistence=persistence_stats['total'],
            max_persistence=persistence_stats['max'],
            num_features=num_features,
            barcode_stats=persistence_stats,
        )
    
    def _compute_betti_numbers(self, complex: SimplicialComplex) -> Dict[int, int]:
        """Compute Betti numbers for all dimensions."""
        betti = {}
        
        # Use persistent homology to compute Betti numbers
        ph = PersistentHomology(complex)
        
        for dim in range(self.max_dimension + 1):
            try:
                # Betti number is the rank of homology group
                homology = ph.compute_homology(dim)
                betti[dim] = len(homology) if homology else 0
            except:
                betti[dim] = 0
                
        return betti
    
    def _create_filtration(self, complex: SimplicialComplex) -> Filtration:
        """
        Create filtration from complex for persistence computation.
        
        Uses simplex dimension as filtration value.
        """
        filtration = Filtration()
        
        # Group simplices by dimension
        by_dimension = defaultdict(list)
        for simplex in complex:
            dim = len(simplex) - 1
            by_dimension[dim].append(simplex)
        
        # Add levels in order of dimension
        for dim in sorted(by_dimension.keys()):
            filtration.add_level(dim, by_dimension[dim])
            
        return filtration
    
    def _compute_persistence_stats(self, barcodes: Dict[int, List]) -> Dict[str, float]:
        """
        Compute statistics from persistence barcodes.
        
        Args:
            barcodes: Dictionary mapping dimension to list of intervals
            
        Returns:
            Dictionary of persistence statistics
        """
        all_persistences = []
        
        for dim, intervals in barcodes.items():
            for interval in intervals:
                if hasattr(interval, 'birth') and hasattr(interval, 'death'):
                    birth = interval.birth
                    death = interval.death if interval.death is not None else float('inf')
                elif isinstance(interval, tuple):
                    birth, death = interval
                    death = death if death is not None else float('inf')
                else:
                    continue
                    
                if death != float('inf'):
                    persistence = death - birth
                    all_persistences.append(persistence)
        
        if not all_persistences:
            return {
                'entropy': 0.0,
                'total': 0.0,
                'max': 0.0,
                'mean': 0.0,
                'std': 0.0,
                'median': 0.0,
            }
        
        persistences = np.array(all_persistences)
        
        # Compute entropy
        if len(persistences) > 0:
            # Normalize to probability distribution
            probs = persistences / persistences.sum()
            entropy = -np.sum(probs * np.log(probs + 1e-10))
        else:
            entropy = 0.0
        
        return {
            'entropy': float(entropy),
            'total': float(persistences.sum()),
            'max': float(persistences.max()),
            'mean': float(persistences.mean()),
            'std': float(persistences.std()),
            'median': float(np.median(persistences)),
        }
    
    def compute_fragmentation(
        self, 
        complex: SimplicialComplex, 
        signature: TopologicalSignature
    ) -> float:
        """
        Compute narrative fragmentation score.
        
        High fragmentation indicates disconnected narrative structure,
        often associated with fabricated content.
        
        Args:
            complex: Simplicial complex
            signature: Topological signature
            
        Returns:
            Fragmentation score in [0, 1]
        """
        # Number of connected components (Betti-0)
        num_components = signature.betti_numbers.get(0, 1)
        
        # Total number of vertices
        vertices = set()
        for simplex in complex:
            vertices.update(simplex)
        num_vertices = len(vertices)
        
        if num_vertices == 0:
            return 0.0
        
        # Fragmentation is ratio of components to vertices
        fragmentation = num_components / num_vertices
        
        # Normalize to [0, 1]
        return min(fragmentation, 1.0)
    
    def compute_coherence(
        self, 
        complex: SimplicialComplex, 
        signature: TopologicalSignature
    ) -> float:
        """
        Compute narrative coherence score.
        
        High coherence indicates well-connected narrative structure.
        
        Args:
            complex: Simplicial complex
            signature: Topological signature
            
        Returns:
            Coherence score in [0, 1]
        """
        # Count edges (1-simplices)
        num_edges = sum(1 for s in complex if len(s) == 2)
        
        # Count vertices
        vertices = set()
        for simplex in complex:
            vertices.update(simplex)
        num_vertices = len(vertices)
        
        if num_vertices <= 1:
            return 0.0
        
        # Maximum possible edges in complete graph
        max_edges = num_vertices * (num_vertices - 1) / 2
        
        # Coherence is edge density
        coherence = num_edges / max_edges if max_edges > 0 else 0.0
        
        return min(coherence, 1.0)
    
    def compute_complexity(self, signature: TopologicalSignature) -> float:
        """
        Compute topological complexity score.
        
        Args:
            signature: Topological signature
            
        Returns:
            Complexity score in [0, 1]
        """
        # Complexity based on number of features and persistence entropy
        total_features = sum(signature.num_features.values())
        entropy = signature.persistence_entropy
        
        # Normalize
        feature_complexity = min(total_features / 100.0, 1.0)
        entropy_complexity = min(entropy / 5.0, 1.0)
        
        # Weighted combination
        complexity = 0.6 * feature_complexity + 0.4 * entropy_complexity
        
        return complexity
    
    def compute_stability(self, signature: TopologicalSignature) -> float:
        """
        Compute topological stability score.
        
        High stability indicates persistent topological features,
        suggesting coherent narrative structure.
        
        Args:
            signature: Topological signature
            
        Returns:
            Stability score in [0, 1]
        """
        # Stability based on persistence statistics
        max_pers = signature.max_persistence
        mean_pers = signature.barcode_stats.get('mean', 0.0)
        
        # Normalize
        if max_pers > 0:
            stability = mean_pers / max_pers
        else:
            stability = 0.0
        
        return min(stability, 1.0)
    
    def generate_shape_code(
        self, 
        signature: TopologicalSignature,
        fragmentation: float,
        coherence: float
    ) -> str:
        """
        Generate human-readable shape code describing topological pattern.
        
        Args:
            signature: Topological signature
            fragmentation: Fragmentation score
            coherence: Coherence score
            
        Returns:
            Shape code string (e.g., 'multi_loop_fragile_burst')
        """
        components = []
        
        # Component structure
        b0 = signature.betti_numbers.get(0, 0)
        if b0 == 1:
            components.append('single')
        elif b0 <= 3:
            components.append('few')
        else:
            components.append('multi')
        
        # Loop structure
        b1 = signature.betti_numbers.get(1, 0)
        if b1 == 0:
            components.append('tree')
        elif b1 == 1:
            components.append('loop')
        else:
            components.append('mesh')
        
        # Stability
        if signature.max_persistence < 1.0:
            components.append('fragile')
        elif signature.max_persistence < 3.0:
            components.append('stable')
        else:
            components.append('robust')
        
        # Temporal pattern
        if signature.persistence_entropy > 2.0:
            components.append('burst')
        elif signature.persistence_entropy > 1.0:
            components.append('evolve')
        else:
            components.append('static')
        
        return '_'.join(components)
