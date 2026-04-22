"""
Topology Statistics Module

Computes statistical properties of simplicial complexes.

Author: Alireza Pourmoslemi
Email: apmath99@gmail.com
"""

import numpy as np
from typing import Dict, List
from collections import defaultdict

from verilogos.core.topology import SimplicialComplex


class TopologyStatistics:
    """
    Computes statistical properties of simplicial complexes.
    
    Statistics computed:
        - Simplex counts by dimension
        - Density
        - Clustering coefficient
        - Degree distribution
        - Connected components
    
    Example:
        >>> stats = TopologyStatistics()
        >>> properties = stats.compute(complex)
    """
    
    def compute(self, complex: SimplicialComplex) -> Dict[str, any]:
        """
        Compute all statistics.
        
        Args:
            complex: SimplicialComplex to analyze
            
        Returns:
            Dictionary of statistics
        """
        stats = {}
        
        # Basic counts
        stats.update(self._count_simplices(complex))
        
        # Density
        stats['density'] = self._compute_density(complex)
        
        # Degree statistics
        stats.update(self._compute_degree_stats(complex))
        
        # Clustering
        stats['clustering_coefficient'] = self._compute_clustering(complex)
        
        return stats
    
    def _count_simplices(self, complex: SimplicialComplex) -> Dict[str, int]:
        """Count simplices by dimension."""
        counts = defaultdict(int)
        
        for simplex in complex:
            dim = len(simplex) - 1
            counts[f'num_simplices_dim{dim}'] += 1
        
        return dict(counts)
    
    def _compute_density(self, complex: SimplicialComplex) -> float:
        """Compute edge density."""
        vertices = set()
        edges = 0
        
        for simplex in complex:
            if len(simplex) == 1:
                vertices.update(simplex)
            elif len(simplex) == 2:
                edges += 1
                vertices.update(simplex)
        
        n = len(vertices)
        if n <= 1:
            return 0.0
        
        max_edges = n * (n - 1) / 2
        return edges / max_edges if max_edges > 0 else 0.0
    
    def _compute_degree_stats(self, complex: SimplicialComplex) -> Dict[str, float]:
        """Compute degree distribution statistics."""
        degrees = defaultdict(int)
        
        for simplex in complex:
            if len(simplex) == 2:
                for vertex in simplex:
                    degrees[vertex] += 1
        
        if not degrees:
            return {
                'mean_degree': 0.0,
                'max_degree': 0.0,
                'min_degree': 0.0,
                'std_degree': 0.0,
            }
        
        degree_values = list(degrees.values())
        
        return {
            'mean_degree': float(np.mean(degree_values)),
            'max_degree': float(np.max(degree_values)),
            'min_degree': float(np.min(degree_values)),
            'std_degree': float(np.std(degree_values)),
        }
    
    def _compute_clustering(self, complex: SimplicialComplex) -> float:
        """Compute global clustering coefficient."""
        # Build adjacency
        adjacency = defaultdict(set)
        for simplex in complex:
            if len(simplex) == 2:
                v1, v2 = simplex
                adjacency[v1].add(v2)
                adjacency[v2].add(v1)
        
        # Count triangles
        triangles = 0
        for simplex in complex:
            if len(simplex) == 3:
                triangles += 1
        
        # Count connected triples
        triples = 0
        for vertex, neighbors in adjacency.items():
            k = len(neighbors)
            if k >= 2:
                triples += k * (k - 1) / 2
        
        if triples == 0:
            return 0.0
        
        return triangles / triples
