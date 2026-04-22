"""
LIAR Dataset Loader

Loads LIAR dataset for fake news detection.

Author: Alireza Pourmoslemi
Email: apmath99@gmail.com
"""

import csv
from pathlib import Path
from typing import Tuple, List

from verilogos.experiments.datasets.base import BaseDataset, DataSample


class LIARDataset(BaseDataset):
    """
    LIAR dataset loader.
    
    The LIAR dataset contains short statements labeled with truthfulness ratings.
    
    Example:
        >>> dataset = LIARDataset()
        >>> dataset.load("data/liar_train.tsv")
        >>> train, val, test = dataset.get_splits()
    """
    
    def __init__(self):
        """Initialize LIAR dataset."""
        super().__init__(name="LIAR")
    
    def load(self, data_path: str, **kwargs) -> bool:
        """
        Load LIAR dataset.
        
        Args:
            data_path: Path to TSV file
            **kwargs: Additional parameters
        
        Returns:
            True if successful
        """
        data_path = Path(data_path)
        
        if not data_path.exists():
            print(f"Warning: Dataset file not found: {data_path}")
            return False
        
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f, delimiter='\t')
                
                for row in reader:
                    if len(row) < 2:
                        continue
                    
                    # LIAR format: label, statement, subject, speaker, ...
                    label_str = row[0].strip().lower()
                    text = row[1].strip() if len(row) > 1 else ""
                    
                    if not text:
                        continue
                    
                    # Convert 6-way labels to binary
                    # pants-fire, false, barely-true -> fake (1)
                    # half-true, mostly-true, true -> real (0)
                    if label_str in ['pants-fire', 'false', 'barely-true']:
                        label = 1
                    elif label_str in ['half-true', 'mostly-true', 'true']:
                        label = 0
                    else:
                        continue
                    
                    # Extract metadata
                    metadata = {
                        'original_label': label_str,
                        'subject': row[2] if len(row) > 2 else '',
                        'speaker': row[3] if len(row) > 3 else ''
                    }
                    
                    sample = DataSample(
                        text=text,
                        label=label,
                        metadata=metadata
                    )
                    
                    self.data.append(sample)
            
            self.is_loaded = True
            print(f"Loaded {len(self.data)} samples from {data_path}")
            return True
        
        except Exception as e:
            print(f"Error loading LIAR dataset: {e}")
            return False
    
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
