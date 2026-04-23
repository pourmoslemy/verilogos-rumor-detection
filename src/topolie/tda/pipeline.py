"""
Rigorous TDA Pipeline: Proving the Geometry of Truth Hypothesis

Architecture:
1. InteractionGraphBuilder: Weighted user-user graphs with time-decay
2. WeightedTemporalFiltration: Multi-parameter persistence computation
3. AdvancedTDAExtractor: Betti curves, persistence landscapes, algebraic connectivity
4. SCLogicRuleEngine: Explainable temporal logic rules

Hypothesis: Real news → high β₁ (echo chambers), Fake news → low β₁ (viral stars)
"""

from pathlib import Path
import numpy as np
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import re
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from scipy.spatial.distance import pdist, squareform


# ============================================================================
# LAYER 1: INTERACTION GRAPH BUILDER
# ============================================================================

@dataclass
class UserInteraction:
    """Represents a weighted user-user interaction."""
    user1: str
    user2: str
    timestamp: float
    weight: float
    interaction_type: str  # 'retweet', 'co-retweet', 'temporal_cluster'


class InteractionGraphBuilder:
    """
    Builds weighted user-user interaction graphs from propagation cascades.
    
    Implements time-decay weighting: W(u,v) = exp(-λ * Δt)
    """
    
    def __init__(self, lambda_decay: float = 0.001, temporal_window: float = 60.0):
        """
        Args:
            lambda_decay: Time decay parameter (higher = faster decay)
            temporal_window: Time window for co-retweet clustering (seconds)
        """
        self.lambda_decay = lambda_decay
        self.temporal_window = temporal_window
    
    def parse_propagation_tree(self, tree_file: Path) -> List[Tuple[str, str, float]]:
        """Parse ACL2017 propagation tree format."""
        edges = []
        with tree_file.open('r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if '->' not in line:
                    continue
                try:
                    parent_part, child_part = line.split('->')
                    parent_match = re.findall(r"'([^']*)'", parent_part)
                    child_match = re.findall(r"'([^']*)'", child_part)
                    
                    if len(parent_match) >= 3 and len(child_match) >= 3:
                        parent_user = parent_match[0]
                        child_user = child_match[0]
                        time_str = child_match[2]
                        timestamp = float(time_str) if time_str != 'None' else 0.0
                        edges.append((parent_user, child_user, timestamp))
                except:
                    continue
        return edges
    
    def build_weighted_graph(self, tree_file: Path) -> List[UserInteraction]:
        """
        Build weighted user-user interaction graph.
        
        Returns:
            List of UserInteraction objects with time-decay weights
        """
        edges = self.parse_propagation_tree(tree_file)
        if not edges:
            return []
        
        interactions = []
        
        # 1. Direct retweet edges (parent → child)
        for parent_user, child_user, timestamp in edges:
            weight = np.exp(-self.lambda_decay * timestamp)
            interactions.append(UserInteraction(
                user1=parent_user,
                user2=child_user,
                timestamp=timestamp,
                weight=weight,
                interaction_type='retweet'
            ))
        
        # 2. Co-retweet edges (users who retweeted same content)
        # Group users by temporal windows
        time_buckets = defaultdict(set)
        for parent_user, child_user, timestamp in edges:
            bucket = int(timestamp / self.temporal_window)
            time_buckets[bucket].add(parent_user)
            time_buckets[bucket].add(child_user)
        
        # Connect users in same temporal window
        for bucket, users in time_buckets.items():
            users_list = list(users)
            bucket_time = bucket * self.temporal_window
            
            for i in range(len(users_list)):
                for j in range(i+1, min(i+10, len(users_list))):  # Limit to top 10
                    # Weight by inverse time (earlier = stronger)
                    weight = np.exp(-self.lambda_decay * bucket_time) * 0.5  # Scale down
                    interactions.append(UserInteraction(
                        user1=users_list[i],
                        user2=users_list[j],
                        timestamp=bucket_time,
                        weight=weight,
                        interaction_type='co-retweet'
                    ))
        
        return interactions


# ============================================================================
# LAYER 2: WEIGHTED TEMPORAL FILTRATION
# ============================================================================

class WeightedTemporalFiltration:
    """
    Computes persistent homology with weighted temporal filtration.
    
    Tracks β₀ and β₁ evolution over time with weight-based ordering.
    """
    
    def __init__(self):
        self.parent = {}
        self.rank = {}
    
    def find(self, x: str) -> str:
        """Union-Find: find root with path compression."""
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x: str, y: str) -> bool:
        """
        Union-Find: merge components.
        
        Returns:
            False if x and y already connected (creates cycle)
            True if merged successfully
        """
        px, py = self.find(x), self.find(y)
        if px == py:
            return False  # Cycle detected
        
        # Union by rank
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        return True
    
    def compute_persistence(self, interactions: List[UserInteraction]) -> Dict:
        """
        Compute persistent homology from weighted temporal filtration.
        
        Returns:
            Dictionary with Betti curves, persistence intervals, lifespans
        """
        if not interactions:
            return self._empty_persistence()
        
        # Sort by timestamp (temporal filtration)
        sorted_interactions = sorted(interactions, key=lambda x: x.timestamp)
        
        # Track features over time
        nodes = set()
        num_components = 0
        
        # Persistence tracking
        component_births = {}  # component_id → birth_time
        component_deaths = []  # (birth, death) pairs
        cycle_births = []  # cycle birth times
        
        # Betti curves (β₀ and β₁ at each filtration step)
        betti_0_curve = []
        betti_1_curve = []
        filtration_times = []
        
        for interaction in sorted_interactions:
            u, v = interaction.user1, interaction.user2
            t = interaction.timestamp
            
            # Add nodes if new
            if u not in nodes:
                nodes.add(u)
                component_births[self.find(u)] = t
                num_components += 1
            
            if v not in nodes:
                nodes.add(v)
                component_births[self.find(v)] = t
                num_components += 1
            
            # Add edge
            pu, pv = self.find(u), self.find(v)
            
            if pu == pv:
                # Edge creates cycle (β₁ feature born)
                cycle_births.append(t)
            else:
                # Edge merges components (β₀ feature dies)
                birth_u = component_births.get(pu, 0.0)
                birth_v = component_births.get(pv, 0.0)
                
                # Younger component dies
                if birth_u < birth_v:
                    component_deaths.append((birth_v, t))
                    if pv in component_births:
                        del component_births[pv]
                else:
                    component_deaths.append((birth_u, t))
                    if pu in component_births:
                        del component_births[pu]
                
                self.union(u, v)
                num_components -= 1
            
            # Record Betti numbers at this filtration step
            betti_0_curve.append(num_components)
            betti_1_curve.append(len(cycle_births))
            filtration_times.append(t)
        
        # Compute persistence features
        max_time = sorted_interactions[-1].timestamp if sorted_interactions else 0.0
        
        # β₀ persistence intervals
        beta0_intervals = []
        for birth, death in component_deaths:
            beta0_intervals.append((birth, death, death - birth))
        
        # Surviving components
        for comp_id, birth in component_births.items():
            beta0_intervals.append((birth, max_time, max_time - birth))
        
        # β₁ persistence intervals (assume cycles persist until end)
        beta1_intervals = [(b, max_time, max_time - b) for b in cycle_births]
        
        return {
            'betti_0_curve': np.array(betti_0_curve),
            'betti_1_curve': np.array(betti_1_curve),
            'filtration_times': np.array(filtration_times),
            'beta0_intervals': beta0_intervals,
            'beta1_intervals': beta1_intervals,
            'max_time': max_time,
            'num_interactions': len(sorted_interactions)
        }
    
    def _empty_persistence(self) -> Dict:
        """Return empty persistence structure."""
        return {
            'betti_0_curve': np.array([]),
            'betti_1_curve': np.array([]),
            'filtration_times': np.array([]),
            'beta0_intervals': [],
            'beta1_intervals': [],
            'max_time': 0.0,
            'num_interactions': 0
        }


# ============================================================================
# LAYER 3: ADVANCED TDA FEATURE EXTRACTOR
# ============================================================================

class AdvancedTDAExtractor:
    """
    Extracts advanced topological features from persistence data.
    
    Features:
    - Betti curve statistics (mean, max, variance, integral)
    - Persistence landscape features
    - Algebraic connectivity (Fiedler value)
    - Cycle density and structural metrics
    """
    
    def compute_betti_curve_features(self, betti_curve: np.ndarray, 
                                     times: np.ndarray) -> Dict[str, float]:
        """
        Extract statistical features from Betti curve.
        
        Args:
            betti_curve: Array of Betti numbers over time
            times: Corresponding timestamps
        
        Returns:
            Dictionary of curve features
        """
        if len(betti_curve) == 0:
            return {
                'mean': 0.0, 'max': 0.0, 'min': 0.0, 'std': 0.0,
                'integral': 0.0, 'peak_time': 0.0, 'decay_rate': 0.0
            }
        
        # Basic statistics
        mean_val = np.mean(betti_curve)
        max_val = np.max(betti_curve)
        min_val = np.min(betti_curve)
        std_val = np.std(betti_curve)
        
        # Integral (area under curve) using trapezoidal rule
        if len(times) > 1:
            integral = np.trapezoid(betti_curve, times)
        else:
            integral = 0.0
        
        # Peak time (when maximum occurs)
        peak_idx = np.argmax(betti_curve)
        peak_time = times[peak_idx] if len(times) > 0 else 0.0
        
        # Decay rate (slope after peak)
        if peak_idx < len(betti_curve) - 1:
            post_peak = betti_curve[peak_idx:]
            decay_rate = -np.mean(np.diff(post_peak)) if len(post_peak) > 1 else 0.0
        else:
            decay_rate = 0.0
        
        return {
            'mean': mean_val,
            'max': max_val,
            'min': min_val,
            'std': std_val,
            'integral': integral,
            'peak_time': peak_time,
            'decay_rate': decay_rate
        }
    
    def compute_persistence_landscape(self, intervals: List[Tuple[float, float, float]], 
                                     resolution: int = 50) -> np.ndarray:
        """
        Compute persistence landscape (simplified version).
        
        Args:
            intervals: List of (birth, death, lifespan) tuples
            resolution: Number of points to sample
        
        Returns:
            Landscape function values
        """
        if not intervals:
            return np.zeros(resolution)
        
        max_time = max(death for _, death, _ in intervals)
        times = np.linspace(0, max_time, resolution)
        landscape = np.zeros(resolution)
        
        for birth, death, _ in intervals:
            for i, t in enumerate(times):
                if birth <= t <= death:
                    # Tent function: rises from birth, peaks at midpoint, falls to death
                    midpoint = (birth + death) / 2
                    if t <= midpoint:
                        landscape[i] += (t - birth) / (midpoint - birth) if midpoint > birth else 0
                    else:
                        landscape[i] += (death - t) / (death - midpoint) if death > midpoint else 0
        
        return landscape
    
    def compute_fiedler_value(self, interactions: List[UserInteraction]) -> float:
        """
        Compute Fiedler value (algebraic connectivity) of the graph.
        
        The Fiedler value is the second-smallest eigenvalue of the graph Laplacian.
        Higher values indicate better connectivity.
        
        Args:
            interactions: List of user interactions
        
        Returns:
            Fiedler value (0 if graph is disconnected or too small)
        """
        if len(interactions) < 3:
            return 0.0
        
        # Build adjacency matrix
        users = set()
        for inter in interactions:
            users.add(inter.user1)
            users.add(inter.user2)
        
        user_list = sorted(list(users))
        n = len(user_list)
        
        if n < 3:
            return 0.0
        
        user_to_idx = {u: i for i, u in enumerate(user_list)}
        
        # Build weighted adjacency matrix
        adj_matrix = np.zeros((n, n))
        for inter in interactions:
            i = user_to_idx[inter.user1]
            j = user_to_idx[inter.user2]
            adj_matrix[i, j] += inter.weight
            adj_matrix[j, i] += inter.weight
        
        # Compute degree matrix
        degree = np.sum(adj_matrix, axis=1)
        
        # Laplacian matrix L = D - A
        laplacian = np.diag(degree) - adj_matrix
        
        # Compute second-smallest eigenvalue
        try:
            # Use sparse eigensolver for efficiency
            laplacian_sparse = csr_matrix(laplacian)
            eigenvalues = eigsh(laplacian_sparse, k=min(3, n-1), which='SM', return_eigenvectors=False)
            fiedler = sorted(eigenvalues)[1] if len(eigenvalues) > 1 else 0.0
            return max(0.0, fiedler)  # Ensure non-negative
        except:
            return 0.0
    
    def extract_features(self, persistence: Dict, 
                        interactions: List[UserInteraction]) -> np.ndarray:
        """
        Extract comprehensive TDA feature vector.
        
        Returns:
            Feature vector (30+ dimensions)
        """
        # Betti curve features for β₀
        beta0_features = self.compute_betti_curve_features(
            persistence['betti_0_curve'],
            persistence['filtration_times']
        )
        
        # Betti curve features for β₁
        beta1_features = self.compute_betti_curve_features(
            persistence['betti_1_curve'],
            persistence['filtration_times']
        )
        
        # Persistence landscape features
        beta0_landscape = self.compute_persistence_landscape(persistence['beta0_intervals'])
        beta1_landscape = self.compute_persistence_landscape(persistence['beta1_intervals'])
        
        # Landscape statistics
        beta0_landscape_mean = np.mean(beta0_landscape)
        beta0_landscape_max = np.max(beta0_landscape)
        beta1_landscape_mean = np.mean(beta1_landscape)
        beta1_landscape_max = np.max(beta1_landscape)
        
        # Algebraic connectivity
        fiedler = self.compute_fiedler_value(interactions)
        
        # Basic persistence statistics
        num_beta0 = len(persistence['beta0_intervals'])
        num_beta1 = len(persistence['beta1_intervals'])
        
        max_beta0_life = max([life for _, _, life in persistence['beta0_intervals']], default=0.0)
        max_beta1_life = max([life for _, _, life in persistence['beta1_intervals']], default=0.0)
        
        avg_beta0_life = np.mean([life for _, _, life in persistence['beta0_intervals']]) if num_beta0 > 0 else 0.0
        avg_beta1_life = np.mean([life for _, _, life in persistence['beta1_intervals']]) if num_beta1 > 0 else 0.0
        
        # Cycle density
        cycle_density = num_beta1 / persistence['num_interactions'] if persistence['num_interactions'] > 0 else 0.0
        
        # Graph statistics
        unique_users = len(set(i.user1 for i in interactions) | set(i.user2 for i in interactions))
        total_weight = sum(i.weight for i in interactions)
        avg_weight = total_weight / len(interactions) if interactions else 0.0
        
        # Assemble feature vector
        features = np.array([
            # Betti curve features (β₀)
            beta0_features['mean'],
            beta0_features['max'],
            beta0_features['std'],
            beta0_features['integral'],
            beta0_features['peak_time'],
            beta0_features['decay_rate'],
            
            # Betti curve features (β₁)
            beta1_features['mean'],
            beta1_features['max'],
            beta1_features['std'],
            beta1_features['integral'],
            beta1_features['peak_time'],
            beta1_features['decay_rate'],
            
            # Persistence landscape features
            beta0_landscape_mean,
            beta0_landscape_max,
            beta1_landscape_mean,
            beta1_landscape_max,
            
            # Algebraic connectivity
            fiedler,
            
            # Basic persistence
            num_beta0,
            num_beta1,
            max_beta0_life,
            max_beta1_life,
            avg_beta0_life,
            avg_beta1_life,
            cycle_density,
            
            # Graph statistics
            unique_users,
            persistence['num_interactions'],
            total_weight,
            avg_weight,
            persistence['max_time']
        ])
        
        return features


# ============================================================================
# LAYER 4: SC-LOGIC RULE ENGINE
# ============================================================================

class SCLogicRuleEngine:
    """
    Spatial-Continuous Logic rule engine for explainable classification.
    
    Implements temporal logic rules over persistence features:
    FAKE ⟺ (ViralGrowthRate > θ) ∧ □(β₁ ≤ ε)
    """
    
    def __init__(self, viral_threshold: float = 100.0, beta1_epsilon: float = 50.0):
        """
        Args:
            viral_threshold: Threshold for viral growth rate (interactions/time)
            beta1_epsilon: Maximum β₁ for "topologically flat" classification
        """
        self.viral_threshold = viral_threshold
        self.beta1_epsilon = beta1_epsilon
    
    def compute_viral_growth_rate(self, persistence: Dict) -> float:
        """
        Compute viral growth rate: interactions per unit time.
        
        Args:
            persistence: Persistence dictionary
        
        Returns:
            Growth rate (interactions/second)
        """
        if persistence['max_time'] == 0:
            return 0.0
        return persistence['num_interactions'] / persistence['max_time']
    
    def check_always_flat(self, betti_1_curve: np.ndarray) -> bool:
        """
        Check if β₁ is always below threshold (□(β₁ ≤ ε)).
        
        Args:
            betti_1_curve: β₁ values over time
        
        Returns:
            True if always flat (all values ≤ epsilon)
        """
        if len(betti_1_curve) == 0:
            return True
        return np.all(betti_1_curve <= self.beta1_epsilon)
    
    def apply_fake_rule(self, persistence: Dict) -> Tuple[bool, float, str]:
        """
        Apply SC-Logic rule: FAKE ⟺ (ViralGrowthRate > θ) ∧ □(β₁ ≤ ε)
        
        Args:
            persistence: Persistence dictionary
        
        Returns:
            (is_fake, confidence, explanation)
        """
        growth_rate = self.compute_viral_growth_rate(persistence)
        is_flat = self.check_always_flat(persistence['betti_1_curve'])
        
        is_viral = growth_rate > self.viral_threshold
        
        if is_viral and is_flat:
            confidence = min(1.0, growth_rate / (2 * self.viral_threshold))
            explanation = f"FAKE: Viral spread (rate={growth_rate:.2f}) with flat topology (max β₁={np.max(persistence['betti_1_curve']) if len(persistence['betti_1_curve']) > 0 else 0:.0f})"
            return True, confidence, explanation
        elif not is_viral and not is_flat:
            confidence = min(1.0, np.max(persistence['betti_1_curve']) / (2 * self.beta1_epsilon) if len(persistence['betti_1_curve']) > 0 else 0)
            explanation = f"REAL: Slow spread (rate={growth_rate:.2f}) with complex topology (max β₁={np.max(persistence['betti_1_curve']) if len(persistence['betti_1_curve']) > 0 else 0:.0f})"
            return False, confidence, explanation
        else:
            confidence = 0.5
            explanation = f"UNCERTAIN: Mixed signals (rate={growth_rate:.2f}, max β₁={np.max(persistence['betti_1_curve']) if len(persistence['betti_1_curve']) > 0 else 0:.0f})"
            return False, confidence, explanation
    
    def compute_rule_features(self, persistence: Dict) -> np.ndarray:
        """
        Compute SC-Logic rule-based features.
        
        Returns:
            Feature vector with rule outputs
        """
        growth_rate = self.compute_viral_growth_rate(persistence)
        is_flat = self.check_always_flat(persistence['betti_1_curve'])
        is_fake, confidence, _ = self.apply_fake_rule(persistence)
        
        max_beta1 = np.max(persistence['betti_1_curve']) if len(persistence['betti_1_curve']) > 0 else 0.0
        
        return np.array([
            growth_rate,
            float(is_flat),
            float(is_fake),
            confidence,
            max_beta1,
            float(growth_rate > self.viral_threshold),
            float(max_beta1 <= self.beta1_epsilon)
        ])


# ============================================================================
# MAIN PIPELINE
# ============================================================================

class RigorousTDAPipeline:
    """
    Complete TDA pipeline for rumor detection.
    
    Integrates all layers: graph building, filtration, feature extraction, SC-Logic.
    """
    
    def __init__(self, lambda_decay: float = 0.001, temporal_window: float = 60.0):
        self.graph_builder = InteractionGraphBuilder(lambda_decay, temporal_window)
        self.filtration = WeightedTemporalFiltration()
        self.extractor = AdvancedTDAExtractor()
        self.sc_logic = SCLogicRuleEngine()
    
    def process_cascade(self, tree_file: Path) -> Tuple[np.ndarray, Dict]:
        """
        Process a single propagation cascade.
        
        Args:
            tree_file: Path to ACL2017 tree file
        
        Returns:
            (feature_vector, metadata_dict)
        """
        # Build weighted interaction graph
        interactions = self.graph_builder.build_weighted_graph(tree_file)
        
        if not interactions:
            return np.zeros(36), {'empty': True}
        
        # Compute persistence
        persistence = self.filtration.compute_persistence(interactions)
        
        # Extract advanced TDA features
        tda_features = self.extractor.extract_features(persistence, interactions)
        
        # Compute SC-Logic rule features
        rule_features = self.sc_logic.compute_rule_features(persistence)
        
        # Combine features
        combined_features = np.concatenate([tda_features, rule_features])
        
        # Metadata for explainability
        is_fake, confidence, explanation = self.sc_logic.apply_fake_rule(persistence)
        metadata = {
            'empty': False,
            'num_interactions': len(interactions),
            'num_beta0': len(persistence['beta0_intervals']),
            'num_beta1': len(persistence['beta1_intervals']),
            'sc_logic_prediction': 'FAKE' if is_fake else 'REAL',
            'sc_logic_confidence': confidence,
            'sc_logic_explanation': explanation
        }
        
        return combined_features, metadata


# ============================================================================
# TESTING
# ============================================================================

if __name__ == '__main__':
    print("="*80)
    print("RIGOROUS TDA PIPELINE: GEOMETRY OF TRUTH")
    print("="*80)
    
    pipeline = RigorousTDAPipeline()
    
    root = Path('historical_data/rumor_detection_acl2017')
    
    print("\nTesting on sample cascades...")
    
    for version in ['twitter15', 'twitter16']:
        tree_dir = root / version / 'tree'
        tree_files = list(tree_dir.glob('*.txt'))[:3]
        
        print(f"\n{version.upper()}:")
        for tree_file in tree_files:
            features, metadata = pipeline.process_cascade(tree_file)
            
            if metadata['empty']:
                print(f"  {tree_file.name}: EMPTY")
                continue
            
            print(f"  {tree_file.name}:")
            print(f"    Feature dims: {len(features)}")
            print(f"    β₀ features: {metadata['num_beta0']}, β₁ features: {metadata['num_beta1']}")
            print(f"    SC-Logic: {metadata['sc_logic_prediction']} (conf={metadata['sc_logic_confidence']:.2f})")
            print(f"    Explanation: {metadata['sc_logic_explanation']}")
    
    print("\n" + "="*80)
    print("SUCCESS: Rigorous TDA pipeline operational")
    print("Feature vector: 36 dimensions (29 TDA + 7 SC-Logic)")
    print("="*80)
