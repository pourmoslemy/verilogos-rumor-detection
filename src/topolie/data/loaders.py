"""
Data Loaders for Hybrid TDA-Text Fake News Detection
Optimized PyTorch Dataset/DataLoader with multiprocessing support
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
from transformers import AutoTokenizer
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

from src.topolie.tda.pipeline import RigorousTDAPipeline


class RumorDataset(Dataset):
    """
    PyTorch Dataset for ACL2017 Rumor Detection with dual modalities.
    
    Supports three modes:
    - 'tda_only': Returns only topological features
    - 'text_only': Returns only text embeddings (tokenized)
    - 'hybrid': Returns both modalities
    """
    
    def __init__(
        self,
        events: List[Dict],
        source_tweets: Dict[str, str],
        mode: str = 'hybrid',
        tokenizer_name: str = 'distilroberta-base',
        max_length: int = 128,
        precomputed_tda: Optional[Dict] = None
    ):
        """
        Args:
            events: List of event dicts with 'event_id', 'label', 'tree_file'
            source_tweets: Dict mapping tweet_id to text content
            mode: 'tda_only', 'text_only', or 'hybrid'
            tokenizer_name: HuggingFace model name for tokenizer
            max_length: Maximum sequence length for text
            precomputed_tda: Optional dict of precomputed TDA features
        """
        assert mode in ['tda_only', 'text_only', 'hybrid'], f"Invalid mode: {mode}"
        
        self.events = events
        self.source_tweets = source_tweets
        self.mode = mode
        self.max_length = max_length
        self.precomputed_tda = precomputed_tda or {}
        
        # Initialize tokenizer for text mode
        if mode in ['text_only', 'hybrid']:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        else:
            self.tokenizer = None
    
    def __len__(self) -> int:
        return len(self.events)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns a dictionary with modality-specific tensors.
        
        Returns:
            {
                'tda_features': torch.Tensor (36,) if mode includes TDA
                'input_ids': torch.Tensor (max_length,) if mode includes text
                'attention_mask': torch.Tensor (max_length,) if mode includes text
                'label': torch.Tensor (1,)
                'event_id': str
            }
        """
        event = self.events[idx]
        event_id = event['event_id']
        label = event['label']
        
        result = {
            'label': torch.tensor(label, dtype=torch.long),
            'event_id': event_id
        }
        
        # TDA features
        if self.mode in ['tda_only', 'hybrid']:
            if event_id in self.precomputed_tda:
                tda_features = self.precomputed_tda[event_id]
            else:
                # Fallback: compute on-the-fly (slow)
                tda_features = np.zeros(36, dtype=np.float32)
            
            result['tda_features'] = torch.tensor(tda_features, dtype=torch.float32)
        
        # Text features
        if self.mode in ['text_only', 'hybrid']:
            # Extract tweet_id from event_id (format: "twitter15_123456")
            tweet_id = event_id.split('_', 1)[1]
            text = self.source_tweets.get(tweet_id, "")
            
            # Tokenize
            encoded = self.tokenizer(
                text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            result['input_ids'] = encoded['input_ids'].squeeze(0)
            result['attention_mask'] = encoded['attention_mask'].squeeze(0)
        
        return result


def _process_single_event_worker(event_data):
    """
    Worker function for parallel TDA extraction (must be at module level for pickling).
    
    Args:
        event_data: Tuple of (event_id, tree_file, lambda_decay, temporal_window)
    
    Returns:
        Tuple of (event_id, features)
    """
    event_id, tree_file, lambda_decay, temporal_window = event_data
    try:
        pipeline = RigorousTDAPipeline(lambda_decay, temporal_window)
        features, metadata = pipeline.process_cascade(Path(tree_file))
        return event_id, features
    except Exception as e:
        # Return zero features on error
        return event_id, np.zeros(36, dtype=np.float32)


def extract_tda_features_parallel(
    events: List[Dict],
    n_workers: int = 4,
    lambda_decay: float = 0.001,
    temporal_window: float = 60.0
) -> Dict[str, np.ndarray]:
    """
    Extract TDA features in parallel using multiprocessing.
    
    Args:
        events: List of event dicts with 'event_id' and 'tree_file'
        n_workers: Number of parallel workers
        lambda_decay: Time decay parameter for TDA pipeline
        temporal_window: Temporal window for co-retweet clustering
    
    Returns:
        Dict mapping event_id to 36-dimensional TDA feature vector
    """
    print(f"Extracting TDA features for {len(events)} events using {n_workers} workers...")
    
    # Prepare work items
    work_items = [(e['event_id'], e['tree_file'], lambda_decay, temporal_window) for e in events]
    
    # Process in parallel
    results = {}
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_process_single_event_worker, item): item for item in work_items}
        
        completed = 0
        for future in as_completed(futures):
            event_id, features = future.result()
            results[event_id] = features
            completed += 1
            
            if completed % 50 == 0:
                print(f"  Processed {completed}/{len(events)} events...")
    
    print(f"Successfully extracted TDA features for {len(results)} events")
    return results


def load_source_tweets(dataset_path: Path) -> Dict[str, str]:
    """
    Load source tweet texts from ACL2017 dataset.
    
    Args:
        dataset_path: Path to twitter15 or twitter16 directory
    
    Returns:
        Dict mapping tweet_id to text content
    """
    source_file = dataset_path / 'source_tweets.txt'
    tweets = {}
    
    with open(source_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                tweet_id = parts[0]
                text = parts[1]
                tweets[tweet_id] = text
    
    return tweets


def create_dataloaders(
    events: List[Dict],
    source_tweets: Dict[str, str],
    mode: str = 'hybrid',
    batch_size: int = 32,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
    precomputed_tda: Optional[Dict] = None,
    tokenizer_name: str = 'distilroberta-base',
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test DataLoaders with stratified splitting.
    
    Args:
        events: List of event dicts
        source_tweets: Dict of tweet texts
        mode: 'tda_only', 'text_only', or 'hybrid'
        batch_size: Batch size for DataLoader
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        random_seed: Random seed for reproducibility
        precomputed_tda: Optional precomputed TDA features
        tokenizer_name: HuggingFace tokenizer name
        num_workers: Number of DataLoader workers
    
    Returns:
        (train_loader, val_loader, test_loader)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    # Stratified split by label
    np.random.seed(random_seed)
    
    real_events = [e for e in events if e['label'] == 1]
    fake_events = [e for e in events if e['label'] == 0]
    
    np.random.shuffle(real_events)
    np.random.shuffle(fake_events)
    
    # Split each class
    def split_events(event_list):
        n = len(event_list)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        
        train = event_list[:n_train]
        val = event_list[n_train:n_train + n_val]
        test = event_list[n_train + n_val:]
        
        return train, val, test
    
    real_train, real_val, real_test = split_events(real_events)
    fake_train, fake_val, fake_test = split_events(fake_events)
    
    train_events = real_train + fake_train
    val_events = real_val + fake_val
    test_events = real_test + fake_test
    
    np.random.shuffle(train_events)
    np.random.shuffle(val_events)
    np.random.shuffle(test_events)
    
    print(f"\nDataset splits:")
    print(f"  Train: {len(train_events)} events (Real: {len(real_train)}, Fake: {len(fake_train)})")
    print(f"  Val:   {len(val_events)} events (Real: {len(real_val)}, Fake: {len(fake_val)})")
    print(f"  Test:  {len(test_events)} events (Real: {len(real_test)}, Fake: {len(fake_test)})")
    
    # Create datasets
    train_dataset = RumorDataset(train_events, source_tweets, mode, tokenizer_name, precomputed_tda=precomputed_tda)
    val_dataset = RumorDataset(val_events, source_tweets, mode, tokenizer_name, precomputed_tda=precomputed_tda)
    test_dataset = RumorDataset(test_events, source_tweets, mode, tokenizer_name, precomputed_tda=precomputed_tda)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def load_acl2017_dataset(
    base_path: str,
    max_events: Optional[int] = None,
    balance_classes: bool = True
) -> Tuple[List[Dict], Dict[str, str]]:
    """
    Load ACL2017 dataset with optional balancing and size limit.
    
    Args:
        base_path: Path to rumor_detection_acl2017 directory
        max_events: Maximum number of events to load (None = all)
        balance_classes: If True, ensure equal real/fake samples
    
    Returns:
        (events_list, source_tweets_dict)
    """
    print(f"Loading ACL2017 dataset from {base_path}...")
    
    all_events = []
    all_source_tweets = {}
    
    for dataset_name in ['twitter15', 'twitter16']:
        dataset_path = Path(base_path) / dataset_name
        
        # Load labels
        label_file = dataset_path / 'label.txt'
        labels = {}
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split(':')
                if len(parts) != 2:
                    continue
                label, tweet_id = parts
                labels[tweet_id] = 1 if label == 'true' else 0
        
        # Load source tweets
        source_tweets = load_source_tweets(dataset_path)
        all_source_tweets.update(source_tweets)
        
        # Load tree files
        tree_dir = dataset_path / 'tree'
        for tree_file in tree_dir.glob('*.txt'):
            tweet_id = tree_file.stem
            
            if tweet_id not in labels:
                continue
            
            all_events.append({
                'event_id': f"{dataset_name}_{tweet_id}",
                'dataset': dataset_name,
                'tweet_id': tweet_id,
                'label': labels[tweet_id],
                'tree_file': str(tree_file)
            })
    
    # Balance classes if requested
    if balance_classes:
        real_events = [e for e in all_events if e['label'] == 1]
        fake_events = [e for e in all_events if e['label'] == 0]
        
        min_count = min(len(real_events), len(fake_events))
        
        if max_events:
            n_per_class = min(max_events // 2, min_count)
        else:
            n_per_class = min_count
        
        np.random.seed(42)
        selected_real = np.random.choice(len(real_events), n_per_class, replace=False)
        selected_fake = np.random.choice(len(fake_events), n_per_class, replace=False)
        
        all_events = [real_events[i] for i in selected_real] + [fake_events[i] for i in selected_fake]
        np.random.shuffle(all_events)
    elif max_events:
        np.random.seed(42)
        indices = np.random.choice(len(all_events), max_events, replace=False)
        all_events = [all_events[i] for i in indices]
    
    print(f"Loaded {len(all_events)} events")
    print(f"  Real: {sum(1 for e in all_events if e['label'] == 1)}")
    print(f"  Fake: {sum(1 for e in all_events if e['label'] == 0)}")
    
    return all_events, all_source_tweets
