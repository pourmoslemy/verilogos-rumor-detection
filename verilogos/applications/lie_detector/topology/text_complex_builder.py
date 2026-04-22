"""
Text Complex Builder Module

Builds simplicial complexes from text by treating sentences as vertices
and creating edges based on semantic similarity.

Author: Alireza Pourmoslemi
Email: apmath99@gmail.com
"""

import numpy as np
from typing import List, Dict, Tuple, Set
from collections import defaultdict

from verilogos.core.topology import SimplicialComplex
from verilogos.applications.lie_detector.data.preprocessor import TextPreprocessor


class TextComplexBuilder:
    """
    Builds simplicial complexes from text documents.
    
    Algorithm:
        1. Segment text into sentences (0-simplices = vertices)
        2. Compute pairwise sentence similarities
        3. Create edges (1-simplices) for similar sentences
        4. Create triangles (2-simplices) for cliques
    
    Example:
        >>> builder = TextComplexBuilder(similarity_threshold=0.3)
        >>> complex = builder.build_complex(text)
        >>> print(f"Vertices: {len(complex.vertices)}")
    """
    
    def __init__(
        self,
        similarity_threshold: float = 0.3,
        max_dimension: int = 2,
        use_tfidf: bool = False
    ):
        """
        Initialize text complex builder.
        
        Args:
            similarity_threshold: Minimum similarity for edge creation
            max_dimension: Maximum simplex dimension to create
            use_tfidf: Whether to use TF-IDF for similarity (vs Jaccard)
        """
        self.similarity_threshold = similarity_threshold
        self.max_dimension = max_dimension
        self.use_tfidf = use_tfidf
        self.preprocessor = TextPreprocessor()
    
    def build_complex(self, text: str) -> SimplicialComplex:
        """
        Build simplicial complex from text.
        
        Args:
            text: Input text document
            
        Returns:
            SimplicialComplex representing text structure
        """
        # Preprocess text
        preprocessed = self.preprocessor.preprocess(text)
        sentences = preprocessed['sentences']
        
        if len(sentences) < 2:
            # Degenerate case: single sentence
            complex = SimplicialComplex()
            complex.add_simplex((0,))
            return complex
        
        # Compute similarity matrix
        similarity_matrix = self._compute_similarity_matrix(sentences)
        
        # Build complex
        complex = self._build_from_similarity(similarity_matrix)
        
        return complex
    
    def _compute_similarity_matrix(self, sentences: List[str]) -> np.ndarray:
        """
        Compute pairwise sentence similarity matrix.
        
        Args:
            sentences: List of sentences
            
        Returns:
            Similarity matrix of shape (n_sentences, n_sentences)
        """
        n = len(sentences)
        similarity = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                sim = self.preprocessor.compute_similarity(sentences[i], sentences[j])
                similarity[i, j] = sim
                similarity[j, i] = sim
        
        # Diagonal is 1.0
        np.fill_diagonal(similarity, 1.0)
        
        return similarity
    
    def _build_from_similarity(self, similarity_matrix: np.ndarray) -> SimplicialComplex:
        """
        Build complex from similarity matrix.
        
        Args:
            similarity_matrix: Pairwise similarity matrix
            
        Returns:
            SimplicialComplex
        """
        n = similarity_matrix.shape[0]
        complex = SimplicialComplex()
        
        # Add all vertices (0-simplices)
        for i in range(n):
            complex.add_simplex((i,))
        
        # Add edges (1-simplices) for similar sentences
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if similarity_matrix[i, j] >= self.similarity_threshold:
                    complex.add_simplex((i, j))
                    edges.append((i, j))
        
        # Add triangles (2-simplices) if max_dimension >= 2
        if self.max_dimension >= 2:
            self._add_triangles(complex, similarity_matrix, edges)
        
        return complex
    
    def _add_triangles(
        self,
        complex: SimplicialComplex,
        similarity_matrix: np.ndarray,
        edges: List[Tuple[int, int]]
    ):
        """
        Add 2-simplices (triangles) to complex.
        
        A triangle (i, j, k) is added if all three edges exist.
        
        Args:
            complex: SimplicialComplex to modify
            similarity_matrix: Similarity matrix
            edges: List of existing edges
        """
        # Build adjacency list
        adjacency = defaultdict(set)
        for i, j in edges:
            adjacency[i].add(j)
            adjacency[j].add(i)
        
        # Find triangles
        triangles = set()
        for i, j in edges:
            # Find common neighbors
            common = adjacency[i].intersection(adjacency[j])
            for k in common:
                # Create sorted tuple to avoid duplicates
                triangle = tuple(sorted([i, j, k]))
                triangles.add(triangle)
        
        # Add triangles to complex
        for triangle in triangles:
            complex.add_simplex(triangle)
    
    def build_complex_with_entities(
        self,
        text: str,
        entities: List[str]
    ) -> SimplicialComplex:
        """
        Build complex incorporating named entities.
        
        Entities become additional vertices connected to sentences
        that mention them.
        
        Args:
            text: Input text
            entities: List of named entities
            
        Returns:
            SimplicialComplex with entity vertices
        """
        # Build base complex from sentences
        preprocessed = self.preprocessor.preprocess(text)
        sentences = preprocessed['sentences']
        
        n_sentences = len(sentences)
        n_entities = len(entities)
        n_total = n_sentences + n_entities
        
        # Extended similarity matrix
        similarity = np.zeros((n_total, n_total))
        
        # Sentence-sentence similarities
        for i in range(n_sentences):
            for j in range(i + 1, n_sentences):
                sim = self.preprocessor.compute_similarity(sentences[i], sentences[j])
                similarity[i, j] = sim
                similarity[j, i] = sim
        
        # Sentence-entity connections
        for i, sentence in enumerate(sentences):
            for j, entity in enumerate(entities):
                entity_idx = n_sentences + j
                # Check if entity appears in sentence
                if entity.lower() in sentence.lower():
                    similarity[i, entity_idx] = 1.0
                    similarity[entity_idx, i] = 1.0
        
        # Build complex
        complex = self._build_from_similarity(similarity)
        
        return complex
    
    def analyze_structure(self, complex: SimplicialComplex) -> Dict[str, any]:
        """
        Analyze structure of text complex.
        
        Args:
            complex: SimplicialComplex to analyze
            
        Returns:
            Dictionary with structural statistics
        """
        # Count simplices by dimension
        simplex_counts = defaultdict(int)
        for simplex in complex:
            dim = len(simplex) - 1
            simplex_counts[dim] += 1
        
        # Compute density
        n_vertices = simplex_counts[0]
        n_edges = simplex_counts[1]
        max_edges = n_vertices * (n_vertices - 1) / 2 if n_vertices > 1 else 0
        density = n_edges / max_edges if max_edges > 0 else 0.0
        
        return {
            'num_vertices': n_vertices,
            'num_edges': n_edges,
            'num_triangles': simplex_counts[2],
            'density': density,
            'simplex_counts': dict(simplex_counts),
        }
