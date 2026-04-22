"""
Network Complex Builder Module

Builds simplicial complexes from entity networks and knowledge graphs.

Author: Alireza Pourmoslemi
Email: apmath99@gmail.com
"""

import numpy as np
from typing import List, Dict, Tuple, Set
from collections import defaultdict

from verilogos.core.topology import SimplicialComplex


class NetworkComplexBuilder:
    """
    Builds simplicial complexes from entity networks.
    
    Treats entities as vertices and relationships as edges.
    Creates higher-dimensional simplices from cliques.
    
    Example:
        >>> builder = NetworkComplexBuilder()
        >>> entities = ["person1", "person2", "org1"]
        >>> relations = [("person1", "person2"), ("person2", "org1")]
        >>> complex = builder.build_from_relations(entities, relations)
    """
    
    def __init__(self, max_dimension: int = 2):
        """
        Initialize network complex builder.
        
        Args:
            max_dimension: Maximum simplex dimension
        """
        self.max_dimension = max_dimension
    
    def build_from_relations(
        self,
        entities: List[str],
        relations: List[Tuple[str, str]]
    ) -> SimplicialComplex:
        """
        Build complex from entity relations.
        
        Args:
            entities: List of entity names
            relations: List of (entity1, entity2) tuples
            
        Returns:
            SimplicialComplex
        """
        # Create entity to index mapping
        entity_to_idx = {entity: idx for idx, entity in enumerate(entities)}
        
        complex = SimplicialComplex()
        
        # Add vertices
        for idx in range(len(entities)):
            complex.add_simplex((idx,))
        
        # Add edges
        edges = []
        for e1, e2 in relations:
            if e1 in entity_to_idx and e2 in entity_to_idx:
                idx1 = entity_to_idx[e1]
                idx2 = entity_to_idx[e2]
                complex.add_simplex((idx1, idx2))
                edges.append((idx1, idx2))
        
        # Add higher-dimensional simplices
        if self.max_dimension >= 2:
            self._add_cliques(complex, edges)
        
        return complex
    
    def build_from_adjacency(self, adjacency_matrix: np.ndarray) -> SimplicialComplex:
        """
        Build complex from adjacency matrix.
        
        Args:
            adjacency_matrix: Binary adjacency matrix
            
        Returns:
            SimplicialComplex
        """
        n = adjacency_matrix.shape[0]
        complex = SimplicialComplex()
        
        # Add vertices
        for i in range(n):
            complex.add_simplex((i,))
        
        # Add edges
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adjacency_matrix[i, j] > 0:
                    complex.add_simplex((i, j))
                    edges.append((i, j))
        
        # Add cliques
        if self.max_dimension >= 2:
            self._add_cliques(complex, edges)
        
        return complex
    
    def _add_cliques(self, complex: SimplicialComplex, edges: List[Tuple[int, int]]):
        """
        Add maximal cliques as higher-dimensional simplices.
        
        Args:
            complex: SimplicialComplex to modify
            edges: List of edges
        """
        # Build adjacency list
        adjacency = defaultdict(set)
        for i, j in edges:
            adjacency[i].add(j)
            adjacency[j].add(i)
        
        # Find triangles
        triangles = set()
        for i, j in edges:
            common = adjacency[i].intersection(adjacency[j])
            for k in common:
                triangle = tuple(sorted([i, j, k]))
                triangles.add(triangle)
        
        # Add triangles
        for triangle in triangles:
            complex.add_simplex(triangle)
    
    def build_from_cooccurrence(
        self,
        entities: List[str],
        contexts: List[List[str]],
        min_cooccurrence: int = 2
    ) -> SimplicialComplex:
        """
        Build complex from entity co-occurrence in contexts.
        
        Args:
            entities: List of entity names
            contexts: List of contexts (each context is list of entities)
            min_cooccurrence: Minimum co-occurrence count for edge
            
        Returns:
            SimplicialComplex
        """
        entity_to_idx = {entity: idx for idx, entity in enumerate(entities)}
        
        # Count co-occurrences
        cooccurrence = defaultdict(int)
        for context in contexts:
            context_indices = [entity_to_idx[e] for e in context if e in entity_to_idx]
            for i in range(len(context_indices)):
                for j in range(i + 1, len(context_indices)):
                    pair = tuple(sorted([context_indices[i], context_indices[j]]))
                    cooccurrence[pair] += 1
        
        # Build complex
        complex = SimplicialComplex()
        
        # Add vertices
        for idx in range(len(entities)):
            complex.add_simplex((idx,))
        
        # Add edges with sufficient co-occurrence
        edges = []
        for (i, j), count in cooccurrence.items():
            if count >= min_cooccurrence:
                complex.add_simplex((i, j))
                edges.append((i, j))
        
        # Add cliques
        if self.max_dimension >= 2:
            self._add_cliques(complex, edges)
        
        return complex
