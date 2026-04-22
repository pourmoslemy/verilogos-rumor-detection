"""
Base Dataset - Abstract interface for all datasets

Author: Alireza Pourmoslemi
Email: apmath99@gmail.com
"""

from abc import ABC, abstractmethod
from typing import Tuple, List, Dict, Any, Optional
import numpy as np
from dataclasses import dataclass

try:
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


@dataclass
class DataSample:
    """Single data sample."""
    text: str
    label: int
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseDataset(ABC):
    """
    Abstract base class for all datasets.
    
    All dataset loaders must implement this interface.
    """
    
    def __init__(self, name: str):
        """
        Initialize dataset.
        
        Args:
            name: Dataset name
        """
        self.name = name
        self.data = []
        self.train_data = []
        self.val_data = []
        self.test_data = []
        self.is_loaded = False
    
    @abstractmethod
    def load(self, data_path: str, **kwargs) -> bool:
        """
        Load dataset from file.
        
        Args:
            data_path: Path to dataset file
            **kwargs: Additional loading parameters
        
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def get_splits(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_state: int = 42
    ) -> Tuple[List[DataSample], List[DataSample], List[DataSample]]:
        """
        Get train/val/test splits.
        
        Args:
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
            random_state: Random seed
        
        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        pass
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get dataset statistics.
        
        Returns:
            Dictionary with statistics
        """
        if not self.is_loaded:
            return {"error": "Dataset not loaded"}
        
        labels = [sample.label for sample in self.data]
        unique_labels = set(labels)
        
        stats = {
            "name": self.name,
            "total_samples": len(self.data),
            "num_classes": len(unique_labels),
            "class_distribution": {
                label: labels.count(label) for label in unique_labels
            }
        }
        
        if self.train_data:
            stats["train_samples"] = len(self.train_data)
        if self.val_data:
            stats["val_samples"] = len(self.val_data)
        if self.test_data:
            stats["test_samples"] = len(self.test_data)
        
        return stats
    
    def __len__(self) -> int:
        """Get dataset size."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> DataSample:
        """Get sample by index."""
        return self.data[idx]
    
    def _split_data(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_state: int = 42,
        stratify: bool = True
    ):
        """
        Internal method to split data with stratification.
        
        CRITICAL FIX: Uses stratified splitting to maintain class distribution
        across train/val/test sets. This prevents class imbalance issues.
        
        Args:
            train_ratio: Training ratio
            val_ratio: Validation ratio
            test_ratio: Test ratio
            random_state: Random seed
            stratify: Whether to use stratified splitting (recommended)
        """
        # Validate ratios
        total = train_ratio + val_ratio + test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Ratios must sum to 1.0, got {total}")
        
        # Extract labels for stratification
        labels = np.array([sample.label for sample in self.data])
        
        # Check class distribution
        unique, counts = np.unique(labels, return_counts=True)
        print(f"\nDataset class distribution:")
        for label, count in zip(unique, counts):
            print(f"  Class {label}: {count} samples ({100*count/len(labels):.1f}%)")
        
        if stratify and SKLEARN_AVAILABLE:
            # STRATIFIED SPLIT (maintains class distribution)
            # First split: train vs (val + test)
            train_idx, temp_idx = train_test_split(
                np.arange(len(self.data)),
                train_size=train_ratio,
                stratify=labels,
                random_state=random_state
            )
            
            # Second split: val vs test
            temp_labels = labels[temp_idx]
            val_size = val_ratio / (val_ratio + test_ratio)
            val_idx, test_idx = train_test_split(
                temp_idx,
                train_size=val_size,
                stratify=temp_labels,
                random_state=random_state
            )
            
            print("Using STRATIFIED splits (maintains class distribution)")
        else:
            # Fallback: random split (not recommended)
            if stratify and not SKLEARN_AVAILABLE:
                print("Warning: sklearn not available, using random split instead of stratified")
            
            np.random.seed(random_state)
            indices = np.random.permutation(len(self.data))
            
            n_train = int(len(self.data) * train_ratio)
            n_val = int(len(self.data) * val_ratio)
            
            train_idx = indices[:n_train]
            val_idx = indices[n_train:n_train + n_val]
            test_idx = indices[n_train + n_val:]
            
            print("Using RANDOM splits (not stratified)")
        
        # Create splits
        self.train_data = [self.data[i] for i in train_idx]
        self.val_data = [self.data[i] for i in val_idx]
        self.test_data = [self.data[i] for i in test_idx]
        
        # Verify class distribution in splits
        train_labels = [s.label for s in self.train_data]
        test_labels = [s.label for s in self.test_data]
        
        train_unique, train_counts = np.unique(train_labels, return_counts=True)
        test_unique, test_counts = np.unique(test_labels, return_counts=True)
        
        print(f"\nSplit sizes: {len(self.train_data)} train, "
              f"{len(self.val_data)} val, {len(self.test_data)} test")
        
        print(f"Train class distribution:")
        for label, count in zip(train_unique, train_counts):
            print(f"  Class {label}: {count} samples ({100*count/len(train_labels):.1f}%)")
        
        print(f"Test class distribution:")
        for label, count in zip(test_unique, test_counts):
            print(f"  Class {label}: {count} samples ({100*count/len(test_labels):.1f}%)")
