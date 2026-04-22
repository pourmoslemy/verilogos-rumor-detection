"""
Ablation Generator - Automatic ablation study configuration

Generates all combinations of experiment configurations for ablation studies.

Author: Alireza Pourmoslemi
Email: apmath99@gmail.com
"""

import itertools
from typing import Dict, List, Any


class AblationGenerator:
    """
    Automatically generate ablation study configurations.
    
    Uses itertools.product to create all combinations of parameters.
    
    Example:
        >>> generator = AblationGenerator()
        >>> config = {
        ...     "topology": [True, False],
        ...     "bert": [True, False],
        ...     "temporal": [True, False]
        ... }
        >>> ablations = generator.generate(config)
        >>> len(ablations)  # 2^3 = 8 combinations
        8
    """
    
    def __init__(self):
        """Initialize ablation generator."""
        pass
    
    def generate(self, ablation_config: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """
        Generate all combinations of ablation parameters.
        
        Args:
            ablation_config: Dictionary mapping parameter names to lists of values
                            Example: {"topology": [True, False], "bert": [True, False]}
        
        Returns:
            List of configuration dictionaries, one per combination
        """
        if not ablation_config:
            return [{}]
        
        # Extract keys and values
        keys = list(ablation_config.keys())
        value_lists = [ablation_config[key] for key in keys]
        
        # Generate all combinations
        combinations = list(itertools.product(*value_lists))
        
        # Convert to list of dicts
        ablations = []
        for combo in combinations:
            ablation = dict(zip(keys, combo))
            ablations.append(ablation)
        
        print(f"Generated {len(ablations)} ablation configurations")
        
        return ablations
    
    def generate_with_baseline(
        self,
        ablation_config: Dict[str, List[Any]],
        baseline: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Generate ablations with a baseline configuration.
        
        The baseline is always included as the first configuration.
        
        Args:
            ablation_config: Ablation parameters
            baseline: Baseline configuration (always included)
        
        Returns:
            List of configurations with baseline first
        """
        ablations = self.generate(ablation_config)
        
        # Check if baseline is already in ablations
        if baseline in ablations:
            # Move baseline to front
            ablations.remove(baseline)
            ablations.insert(0, baseline)
        else:
            # Add baseline at front
            ablations.insert(0, baseline)
        
        return ablations
    
    def generate_incremental(
        self,
        features: List[str],
        start_with_all: bool = True
    ) -> List[Dict[str, bool]]:
        """
        Generate incremental ablation (add/remove one feature at a time).
        
        Args:
            features: List of feature names
            start_with_all: If True, start with all features enabled.
                           If False, start with all disabled.
        
        Returns:
            List of configurations showing incremental changes
        """
        configs = []
        
        if start_with_all:
            # Start with all features, remove one at a time
            base_config = {f: True for f in features}
            configs.append(base_config.copy())
            
            for feature in features:
                config = base_config.copy()
                config[feature] = False
                configs.append(config)
        else:
            # Start with no features, add one at a time
            base_config = {f: False for f in features}
            configs.append(base_config.copy())
            
            for feature in features:
                config = base_config.copy()
                config[feature] = True
                configs.append(config)
        
        print(f"Generated {len(configs)} incremental ablation configurations")
        
        return configs
    
    def generate_pairwise(
        self,
        features: List[str]
    ) -> List[Dict[str, bool]]:
        """
        Generate pairwise feature combinations.
        
        Useful for studying feature interactions.
        
        Args:
            features: List of feature names
        
        Returns:
            List of configurations with pairs of features enabled
        """
        configs = []
        
        # All features disabled
        configs.append({f: False for f in features})
        
        # Single features
        for feature in features:
            config = {f: False for f in features}
            config[feature] = True
            configs.append(config)
        
        # Pairs of features
        for i, f1 in enumerate(features):
            for f2 in features[i+1:]:
                config = {f: False for f in features}
                config[f1] = True
                config[f2] = True
                configs.append(config)
        
        # All features enabled
        configs.append({f: True for f in features})
        
        print(f"Generated {len(configs)} pairwise ablation configurations")
        
        return configs
    
    def filter_invalid(
        self,
        ablations: List[Dict[str, Any]],
        validator: callable
    ) -> List[Dict[str, Any]]:
        """
        Filter out invalid ablation configurations.
        
        Args:
            ablations: List of ablation configs
            validator: Function that returns True if config is valid
        
        Returns:
            Filtered list of valid configurations
        """
        valid = [a for a in ablations if validator(a)]
        
        removed = len(ablations) - len(valid)
        if removed > 0:
            print(f"Filtered out {removed} invalid configurations")
        
        return valid
    
    def add_metadata(
        self,
        ablations: List[Dict[str, Any]],
        metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Add metadata to all ablation configurations.
        
        Args:
            ablations: List of ablation configs
            metadata: Metadata to add to each config
        
        Returns:
            Ablations with metadata added
        """
        return [
            {**ablation, **metadata}
            for ablation in ablations
        ]
    
    def generate_grid_search(
        self,
        param_grid: Dict[str, List[Any]]
    ) -> List[Dict[str, Any]]:
        """
        Generate grid search configurations (alias for generate).
        
        Args:
            param_grid: Parameter grid
        
        Returns:
            List of all parameter combinations
        """
        return self.generate(param_grid)
    
    def summarize(self, ablations: List[Dict[str, Any]]) -> str:
        """
        Generate a summary of ablation configurations.
        
        Args:
            ablations: List of ablation configs
        
        Returns:
            Summary string
        """
        if not ablations:
            return "No ablations generated"
        
        summary = []
        summary.append(f"Total configurations: {len(ablations)}")
        
        # Count unique values per parameter
        all_keys = set()
        for ablation in ablations:
            all_keys.update(ablation.keys())
        
        for key in sorted(all_keys):
            values = [a.get(key) for a in ablations if key in a]
            unique_values = set(values)
            summary.append(f"  {key}: {len(unique_values)} unique values")
        
        return "\n".join(summary)
