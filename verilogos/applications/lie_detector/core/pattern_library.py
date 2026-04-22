"""
Pattern Library Module

Pre-defined topological patterns and signatures for truth vs. lies.
Based on empirical analysis of fake news topological structures.

Author: Alireza Pourmoslemi
Email: apmath99@gmail.com
"""

from typing import Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum


class PatternType(Enum):
    """Types of topological patterns."""
    TRUTH = "truth"
    LIE = "lie"
    UNCERTAIN = "uncertain"


@dataclass
class TopologicalPattern:
    """
    A pre-defined topological pattern.
    
    Attributes:
        name: Pattern name
        pattern_type: Truth or lie
        description: Human-readable description
        betti_range: Expected Betti number ranges
        fragmentation_range: Expected fragmentation score range
        coherence_range: Expected coherence score range
        shape_codes: Associated shape codes
        confidence: Confidence in pattern (0-1)
    """
    name: str
    pattern_type: PatternType
    description: str
    betti_range: Dict[int, Tuple[float, float]]
    fragmentation_range: Tuple[float, float]
    coherence_range: Tuple[float, float]
    shape_codes: List[str]
    confidence: float = 0.8


class PatternLibrary:
    """
    Library of pre-defined topological patterns.
    
    Contains empirically-derived patterns that distinguish truth from lies
    based on topological structure analysis.
    
    Example:
        >>> library = PatternLibrary()
        >>> pattern = library.match_pattern(truth_shape)
        >>> print(pattern.pattern_type)
    """
    
    def __init__(self):
        """Initialize pattern library with pre-defined patterns."""
        self.patterns = self._initialize_patterns()
    
    def _initialize_patterns(self) -> List[TopologicalPattern]:
        """
        Initialize library with known patterns.
        
        Returns:
            List of TopologicalPattern objects
        """
        patterns = []
        
        # Truth patterns
        patterns.append(TopologicalPattern(
            name="coherent_narrative",
            pattern_type=PatternType.TRUTH,
            description="Single connected component with moderate loops, high coherence",
            betti_range={0: (1, 1), 1: (1, 3), 2: (0, 1)},
            fragmentation_range=(0.0, 0.2),
            coherence_range=(0.6, 1.0),
            shape_codes=["single_loop_stable_evolve", "single_mesh_robust_evolve"],
            confidence=0.85,
        ))
        
        patterns.append(TopologicalPattern(
            name="simple_truth",
            pattern_type=PatternType.TRUTH,
            description="Tree structure, no loops, high stability",
            betti_range={0: (1, 1), 1: (0, 0), 2: (0, 0)},
            fragmentation_range=(0.0, 0.1),
            coherence_range=(0.5, 0.9),
            shape_codes=["single_tree_robust_static", "single_tree_stable_static"],
            confidence=0.9,
        ))
        
        patterns.append(TopologicalPattern(
            name="complex_truth",
            pattern_type=PatternType.TRUTH,
            description="Multiple components but high coherence within each",
            betti_range={0: (2, 4), 1: (2, 5), 2: (0, 2)},
            fragmentation_range=(0.1, 0.3),
            coherence_range=(0.6, 0.9),
            shape_codes=["few_mesh_stable_evolve", "few_loop_robust_evolve"],
            confidence=0.75,
        ))
        
        # Lie patterns
        patterns.append(TopologicalPattern(
            name="fragmented_lie",
            pattern_type=PatternType.LIE,
            description="Many disconnected components, low coherence",
            betti_range={0: (5, 20), 1: (0, 2), 2: (0, 0)},
            fragmentation_range=(0.4, 1.0),
            coherence_range=(0.0, 0.3),
            shape_codes=["multi_tree_fragile_burst", "multi_loop_fragile_burst"],
            confidence=0.88,
        ))
        
        patterns.append(TopologicalPattern(
            name="unstable_lie",
            pattern_type=PatternType.LIE,
            description="Low persistence, fragile features, burst pattern",
            betti_range={0: (2, 10), 1: (1, 5), 2: (0, 1)},
            fragmentation_range=(0.2, 0.6),
            coherence_range=(0.2, 0.5),
            shape_codes=["few_loop_fragile_burst", "multi_mesh_fragile_burst"],
            confidence=0.82,
        ))
        
        patterns.append(TopologicalPattern(
            name="contradictory_lie",
            pattern_type=PatternType.LIE,
            description="High loop count indicating contradictions",
            betti_range={0: (1, 3), 1: (5, 15), 2: (1, 5)},
            fragmentation_range=(0.1, 0.4),
            coherence_range=(0.3, 0.6),
            shape_codes=["single_mesh_fragile_burst", "few_mesh_fragile_evolve"],
            confidence=0.78,
        ))
        
        return patterns
    
    def match_pattern(
        self, 
        truth_shape,
        threshold: float = 0.7
    ) -> Tuple[TopologicalPattern, float]:
        """
        Match truth shape to closest pattern.
        
        Args:
            truth_shape: TruthShape object to match
            threshold: Minimum similarity threshold
            
        Returns:
            Tuple of (best_pattern, similarity_score)
        """
        best_pattern = None
        best_score = 0.0
        
        for pattern in self.patterns:
            score = self._compute_similarity(truth_shape, pattern)
            if score > best_score:
                best_score = score
                best_pattern = pattern
        
        if best_score < threshold:
            # Return uncertain pattern
            return self._get_uncertain_pattern(), best_score
        
        return best_pattern, best_score
    
    def _compute_similarity(self, truth_shape, pattern: TopologicalPattern) -> float:
        """
        Compute similarity between truth shape and pattern.
        
        Args:
            truth_shape: TruthShape object
            pattern: TopologicalPattern to compare
            
        Returns:
            Similarity score in [0, 1]
        """
        scores = []
        
        # Check Betti numbers
        for dim, (min_val, max_val) in pattern.betti_range.items():
            betti = truth_shape.signature.betti_numbers.get(dim, 0)
            if min_val <= betti <= max_val:
                scores.append(1.0)
            else:
                # Penalize based on distance
                if betti < min_val:
                    scores.append(max(0, 1 - (min_val - betti) / min_val))
                else:
                    scores.append(max(0, 1 - (betti - max_val) / max_val))
        
        # Check fragmentation
        frag = truth_shape.fragmentation_score
        frag_min, frag_max = pattern.fragmentation_range
        if frag_min <= frag <= frag_max:
            scores.append(1.0)
        else:
            if frag < frag_min:
                scores.append(max(0, 1 - (frag_min - frag)))
            else:
                scores.append(max(0, 1 - (frag - frag_max)))
        
        # Check coherence
        coh = truth_shape.coherence_score
        coh_min, coh_max = pattern.coherence_range
        if coh_min <= coh <= coh_max:
            scores.append(1.0)
        else:
            if coh < coh_min:
                scores.append(max(0, 1 - (coh_min - coh)))
            else:
                scores.append(max(0, 1 - (coh - coh_max)))
        
        # Check shape code match
        if truth_shape.shape_code in pattern.shape_codes:
            scores.append(1.0)
        else:
            scores.append(0.5)  # Partial credit
        
        # Weighted average
        return sum(scores) / len(scores) if scores else 0.0
    
    def _get_uncertain_pattern(self) -> TopologicalPattern:
        """Return default uncertain pattern."""
        return TopologicalPattern(
            name="uncertain",
            pattern_type=PatternType.UNCERTAIN,
            description="Pattern does not match known signatures",
            betti_range={},
            fragmentation_range=(0.0, 1.0),
            coherence_range=(0.0, 1.0),
            shape_codes=[],
            confidence=0.5,
        )
    
    def get_patterns_by_type(self, pattern_type: PatternType) -> List[TopologicalPattern]:
        """
        Get all patterns of a specific type.
        
        Args:
            pattern_type: Type of patterns to retrieve
            
        Returns:
            List of matching patterns
        """
        return [p for p in self.patterns if p.pattern_type == pattern_type]
    
    def add_pattern(self, pattern: TopologicalPattern):
        """
        Add new pattern to library.
        
        Args:
            pattern: TopologicalPattern to add
        """
        self.patterns.append(pattern)
