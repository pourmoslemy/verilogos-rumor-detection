"""
PHEME Dataset Loader

Loads PHEME dataset for rumor detection.

Author: Alireza Pourmoslemi
Email: apmath99@gmail.com
"""

import json
from pathlib import Path
from typing import Tuple, List

from verilogos.experiments.datasets.base import BaseDataset, DataSample


class PHEMEDataset(BaseDataset):
    """
    PHEME dataset loader.
    
    PHEME is a dataset of rumors and non-rumors from Twitter.
    
    Example:
        >>> dataset = PHEMEDataset()
        >>> dataset.load("data/pheme")
        >>> train, val, test = dataset.get_splits()
    """
    
    def __init__(self):
        """Initialize PHEME dataset."""
        super().__init__(name="PHEME")
    
    def load(self, data_path: str, **kwargs) -> bool:
        """
        Load PHEME dataset.
        
        Args:
            data_path: Path to PHEME directory or JSON file
            **kwargs: Additional parameters
        
        Returns:
            True if successful
        """
        data_path = Path(data_path)
        
        if not data_path.exists():
            print(f"Warning: Dataset path not found: {data_path}")
            return False
        
        try:
            if data_path.is_file():
                # Single JSON file
                self._load_json_file(data_path)
            elif data_path.is_dir():
                # Directory structure
                self._load_directory(data_path)
            else:
                print(f"Invalid path: {data_path}")
                return False
            
            self.is_loaded = True
            print(f"Loaded {len(self.data)} samples from {data_path}")
            return True
        
        except Exception as e:
            print(f"Error loading PHEME dataset: {e}")
            return False
    
    def _load_json_file(self, filepath: Path):
        """Load from single JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict) and 'data' in data:
            items = data['data']
        else:
            items = [data]
        
        for item in items:
            text = item.get('text', item.get('tweet_text', ''))
            label_str = item.get('label', item.get('veracity', ''))
            
            if not text:
                continue
            
            # Convert label
            if label_str in ['rumor', 'false', 'unverified', 1, '1']:
                label = 1
            elif label_str in ['non-rumor', 'true', 'verified', 0, '0']:
                label = 0
            else:
                continue
            
            sample = DataSample(
                text=text,
                label=label,
                metadata={
                    'event': item.get('event', ''),
                    'tweet_id': item.get('tweet_id', '')
                }
            )
            
            self.data.append(sample)
    
    def _load_directory(self, dirpath: Path):
        """Load from directory structure."""
        # Look for JSON files in subdirectories
        for json_file in dirpath.rglob('*.json'):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    item = json.load(f)
                
                text = item.get('text', '')
                if not text:
                    continue
                
                # Determine label from directory structure
                # Typically: rumors/ and non-rumors/
                if 'rumor' in str(json_file.parent).lower():
                    if 'non' in str(json_file.parent).lower():
                        label = 0
                    else:
                        label = 1
                else:
                    # Try to get from file
                    label_str = item.get('label', item.get('veracity', ''))
                    if label_str in ['rumor', 'false', 1]:
                        label = 1
                    elif label_str in ['non-rumor', 'true', 0]:
                        label = 0
                    else:
                        continue
                
                sample = DataSample(
                    text=text,
                    label=label,
                    metadata={
                        'file': str(json_file.name),
                        'event': json_file.parent.name
                    }
                )
                
                self.data.append(sample)
            
            except Exception as e:
                # Skip problematic files
                continue
    
    def get_splits(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_state: int = 42
    ) -> Tuple[List[DataSample], List[DataSample], List[DataSample]]:
        """Get train/val/test splits."""
        if not self.is_loaded:
            raise ValueError("Dataset not loaded. Call load() first.")
        
        self._split_data(train_ratio, val_ratio, test_ratio, random_state)
        
        return self.train_data, self.val_data, self.test_data
