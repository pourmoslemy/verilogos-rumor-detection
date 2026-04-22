"""
Explainability Module

Generates human-readable explanations for predictions.

Author: Alireza Pourmoslemi
Email: apmath99@gmail.com
"""

from typing import Dict, List
import numpy as np

from verilogos.applications.lie_detector.core.truth_geometry import TruthShape
from verilogos.applications.lie_detector.core.pattern_library import PatternLibrary, PatternType


class ExplainabilityEngine:
    """
    Generates explanations for lie detector predictions.
    
    Example:
        >>> engine = ExplainabilityEngine()
        >>> explanation = engine.explain(truth_shape, prediction, confidence)
    """
    
    def __init__(self):
        """Initialize explainability engine."""
        self.pattern_library = PatternLibrary()
    
    def explain(
        self,
        truth_shape: TruthShape,
        prediction: int,
        confidence: float,
        feature_importance: Dict[str, float] = None
    ) -> str:
        """
        Generate explanation for prediction.
        
        Args:
            truth_shape: TruthShape of the article
            prediction: Predicted label (0=fake, 1=real)
            confidence: Prediction confidence
            feature_importance: Feature importances (optional)
            
        Returns:
            Human-readable explanation string
        """
        # Match to pattern
        pattern, similarity = self.pattern_library.match_pattern(truth_shape)
        
        # Build explanation
        explanation_parts = []
        
        # Prediction
        label_text = "REAL" if prediction == 1 else "FAKE"
        explanation_parts.append(f"Prediction: {label_text} (confidence: {confidence:.2%})")
        
        # Pattern match
        explanation_parts.append(f"\nTopological Pattern: {pattern.name}")
        explanation_parts.append(f"Pattern Type: {pattern.pattern_type.value}")
        explanation_parts.append(f"Pattern Match: {similarity:.2%}")
        
        # Shape code
        explanation_parts.append(f"\nShape Code: {truth_shape.shape_code}")
        
        # Topological features
        sig = truth_shape.signature
        explanation_parts.append(f"\nTopological Features:")
        explanation_parts.append(f"  - Connected Components (B0): {sig.betti_numbers.get(0, 0)}")
        explanation_parts.append(f"  - Loops (B1): {sig.betti_numbers.get(1, 0)}")
        explanation_parts.append(f"  - Voids (B2): {sig.betti_numbers.get(2, 0)}")
        explanation_parts.append(f"  - Euler Characteristic: {sig.euler_characteristic}")
        
        # Geometric properties
        explanation_parts.append(f"\nGeometric Properties:")
        explanation_parts.append(f"  - Fragmentation: {truth_shape.fragmentation_score:.2f}")
        explanation_parts.append(f"  - Coherence: {truth_shape.coherence_score:.2f}")
        explanation_parts.append(f"  - Stability: {truth_shape.stability_score:.2f}")
        
        # Interpretation
        explanation_parts.append(f"\nInterpretation:")
        explanation_parts.append(self._generate_interpretation(truth_shape, pattern))
        
        return "\n".join(explanation_parts)
    
    def _generate_interpretation(self, truth_shape: TruthShape, pattern) -> str:
        """Generate interpretation text."""
        interpretations = []
        
        # Fragmentation
        if truth_shape.fragmentation_score > 0.5:
            interpretations.append("High fragmentation suggests disconnected narrative structure.")
        elif truth_shape.fragmentation_score < 0.2:
            interpretations.append("Low fragmentation indicates cohesive narrative.")
        
        # Coherence
        if truth_shape.coherence_score > 0.6:
            interpretations.append("High coherence shows well-connected claims.")
        elif truth_shape.coherence_score < 0.3:
            interpretations.append("Low coherence suggests weak claim connections.")
        
        # Stability
        if truth_shape.stability_score > 0.7:
            interpretations.append("High stability indicates persistent topological features.")
        elif truth_shape.stability_score < 0.3:
            interpretations.append("Low stability suggests fragile, transient features.")
        
        # Pattern-based
        if pattern.pattern_type == PatternType.LIE:
            interpretations.append(f"Pattern matches known misinformation signature: {pattern.description}")
        elif pattern.pattern_type == PatternType.TRUTH:
            interpretations.append(f"Pattern matches truthful content signature: {pattern.description}")
        
        return " ".join(interpretations)
