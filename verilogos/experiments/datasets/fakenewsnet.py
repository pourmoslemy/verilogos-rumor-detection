"""
FakeNewsNet Dataset Loader

Loads FakeNewsNet dataset (Politifact and GossipCop).

Author: Alireza Pourmoslemi
Email: apmath99@gmail.com
"""

import json
import csv
from pathlib import Path
from typing import Tuple, List, Optional

from verilogos.experiments.datasets.base import BaseDataset, DataSample


class FakeNewsNetDataset(BaseDataset):
    """
    FakeNewsNet dataset loader.
    
    Supports both CSV and JSON formats.
    
    Example:
        >>> dataset = FakeNewsNetDataset()
        >>> dataset.load("data/fakenewsnet.csv")
        >>> train, val, test = dataset.get_splits()
    """
    
    def __init__(self):
        """Initialize FakeNewsNet dataset."""
        super().__init__(name="FakeNewsNet")
    
    def load(self, data_path: str, format: str = "auto", **kwargs) -> bool:
        """
        Load FakeNewsNet dataset.
        
        Args:
            data_path: Path to dataset file
            format: File format ('csv', 'json', or 'auto')
            **kwargs: Additional parameters
        
        Returns:
            True if successful
        """
        data_path = Path(data_path)
        
        if not data_path.exists():
            print(f"Warning: Dataset file not found: {data_path}")
            return False
        
        # Auto-detect format
        if format == "auto":
            if data_path.suffix == ".csv":
                format = "csv"
            elif data_path.suffix == ".json":
                format = "json"
            else:
                print(f"Warning: Unknown file format: {data_path.suffix}")
                return False
        
        try:
            if format == "csv":
                self._load_csv(data_path)
            elif format == "json":
                self._load_json(data_path)
            else:
                print(f"Unsupported format: {format}")
                return False
            
            self.is_loaded = True
            print(f"Loaded {len(self.data)} samples from {data_path}")
            return True
        
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return False
    
    def _load_csv(self, filepath: Path):
        """Load from CSV file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                # Extract text and label
                text = row.get('text', row.get('content', ''))
                label_str = row.get('label', row.get('class', ''))
                
                # Convert label to int
                if label_str.lower() in ['fake', 'false', '1']:
                    label = 1
                elif label_str.lower() in ['real', 'true', '0']:
                    label = 0
                else:
                    continue  # Skip invalid labels
                
                # Create sample
                sample = DataSample(
                    text=text,
                    label=label,
                    metadata={
                        'source': row.get('source', 'unknown'),
                        'id': row.get('id', '')
                    }
                )
                
                self.data.append(sample)
    
    def _load_json(self, filepath: Path):
        """Load from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle different JSON structures
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict) and 'data' in data:
            items = data['data']
        else:
            items = [data]
        
        for item in items:
            text = item.get('text', item.get('content', ''))
            label_str = item.get('label', item.get('class', ''))
            
            # Convert label
            if label_str in ['fake', 'false', 1, '1']:
                label = 1
            elif label_str in ['real', 'true', 0, '0']:
                label = 0
            else:
                continue
            
            sample = DataSample(
                text=text,
                label=label,
                metadata={
                    'source': item.get('source', 'unknown'),
                    'id': item.get('id', '')
                }
            )
            
            self.data.append(sample)
    
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
            train_ratio: Training ratio
            val_ratio: Validation ratio
            test_ratio: Test ratio
            random_state: Random seed
        
        Returns:
            Tuple of (train, val, test)
        """
        if not self.is_loaded:
            raise ValueError("Dataset not loaded. Call load() first.")
        
        self._split_data(train_ratio, val_ratio, test_ratio, random_state)
        
        return self.train_data, self.val_data, self.test_data
    
    def create_synthetic(self, n_samples: int = 100) -> bool:
        """
        Create synthetic FakeNewsNet-style data for testing.
        
        Args:
            n_samples: Number of samples to generate
        
        Returns:
            True if successful
        """
        import numpy as np
        
        # Real news templates
        real_templates = [
            "The government announced new economic policies today. Officials stated the measures will take effect next month.",
            "Scientists discovered a breakthrough in renewable energy. The research team published findings in a peer-reviewed journal.",
            "Local authorities reported improved public safety statistics. Crime rates decreased by fifteen percent this quarter.",
            "International trade negotiations concluded successfully. Both nations agreed to reduce tariffs on key goods.",
            "Medical researchers identified a promising treatment approach. Clinical trials showed encouraging results."
        ]
        
        # Fake news templates
        fake_templates = [
            "BREAKING: Secret conspiracy revealed! Anonymous sources claim shocking truth. Mainstream media refuses to report.",
            "You won't believe what happened next! Incredible discovery changes everything. Scientists baffled.",
            "Miracle cure discovered! Doctors hate this one simple trick. Big pharma doesn't want you to know.",
            "Shocking revelation about famous person! Insider leaks explosive information. Career over.",
            "Economic collapse imminent! Experts predict disaster. Stock market manipulation exposed."
        ]
        
        self.data = []
        
        for i in range(n_samples // 2):
            # Real sample
            text = real_templates[i % len(real_templates)]
            self.data.append(DataSample(
                text=text,
                label=0,
                metadata={'source': 'synthetic', 'id': f'real_{i}'}
            ))
            
            # Fake sample
            text = fake_templates[i % len(fake_templates)]
            self.data.append(DataSample(
                text=text,
                label=1,
                metadata={'source': 'synthetic', 'id': f'fake_{i}'}
            ))
        
        # Shuffle
        np.random.shuffle(self.data)
        
        self.is_loaded = True
        print(f"Created {len(self.data)} synthetic samples")
        return True
