"""
Dataset Module

Handles loading and management of fake news datasets.
Supports FakeNewsNet (Politifact) format.

Author: Alireza Pourmoslemi
Email: apmath99@gmail.com
"""

import json
import os
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass
class NewsArticle:
    """
    Represents a single news article.
    
    Attributes:
        id: Unique article identifier
        title: Article title
        text: Article content
        label: Ground truth label (0=fake, 1=real)
        source: Source publication
        date: Publication date
        metadata: Additional metadata
    """
    id: str
    title: str
    text: str
    label: int
    source: str = ""
    date: str = ""
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'title': self.title,
            'text': self.text,
            'label': self.label,
            'source': self.source,
            'date': self.date,
            'metadata': self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NewsArticle':
        """Create from dictionary."""
        return cls(
            id=data.get('id', ''),
            title=data.get('title', ''),
            text=data.get('text', ''),
            label=data.get('label', 0),
            source=data.get('source', ''),
            date=data.get('date', ''),
            metadata=data.get('metadata', {}),
        )


class FakeNewsDataset:
    """
    Dataset class for fake news detection.
    
    Supports loading from JSON files in FakeNewsNet format.
    
    Example:
        >>> dataset = FakeNewsDataset()
        >>> dataset.load_from_json('data/politifact.json')
        >>> train, test = dataset.train_test_split(test_size=0.2)
    """
    
    def __init__(self):
        """Initialize empty dataset."""
        self.articles: List[NewsArticle] = []
        self.label_counts = {0: 0, 1: 0}
        
    def load_from_json(self, filepath: str):
        """
        Load dataset from JSON file.
        
        Expected format:
        [
            {
                "id": "article_id",
                "title": "Article title",
                "text": "Article content",
                "label": 0 or 1,
                "source": "source_name",
                "date": "YYYY-MM-DD"
            },
            ...
        ]
        
        Args:
            filepath: Path to JSON file
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset file not found: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.articles = []
        for item in data:
            article = NewsArticle.from_dict(item)
            self.articles.append(article)
            self.label_counts[article.label] = self.label_counts.get(article.label, 0) + 1
        
        print(f"Loaded {len(self.articles)} articles")
        print(f"Label distribution: {self.label_counts}")
    
    def load_from_directory(self, dirpath: str):
        """
        Load dataset from directory of JSON files.
        
        Args:
            dirpath: Path to directory containing JSON files
        """
        dirpath = Path(dirpath)
        if not dirpath.exists():
            raise FileNotFoundError(f"Directory not found: {dirpath}")
        
        json_files = list(dirpath.glob('*.json'))
        
        for json_file in json_files:
            try:
                self.load_from_json(str(json_file))
            except Exception as e:
                print(f"Error loading {json_file}: {e}")
    
    def add_article(self, article: NewsArticle):
        """
        Add single article to dataset.
        
        Args:
            article: NewsArticle object
        """
        self.articles.append(article)
        self.label_counts[article.label] = self.label_counts.get(article.label, 0) + 1
    
    def get_article(self, article_id: str) -> Optional[NewsArticle]:
        """
        Get article by ID.
        
        Args:
            article_id: Article identifier
            
        Returns:
            NewsArticle if found, None otherwise
        """
        for article in self.articles:
            if article.id == article_id:
                return article
        return None
    
    def filter_by_label(self, label: int) -> 'FakeNewsDataset':
        """
        Create new dataset with only specified label.
        
        Args:
            label: Label to filter (0=fake, 1=real)
            
        Returns:
            New FakeNewsDataset with filtered articles
        """
        filtered = FakeNewsDataset()
        for article in self.articles:
            if article.label == label:
                filtered.add_article(article)
        return filtered
    
    def train_test_split(
        self, 
        test_size: float = 0.2,
        random_state: Optional[int] = None
    ) -> Tuple['FakeNewsDataset', 'FakeNewsDataset']:
        """
        Split dataset into train and test sets.
        
        Args:
            test_size: Fraction of data for test set
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (train_dataset, test_dataset)
        """
        if random_state is not None:
            np.random.seed(random_state)
        
        # Shuffle indices
        indices = np.arange(len(self.articles))
        np.random.shuffle(indices)
        
        # Split
        split_idx = int(len(indices) * (1 - test_size))
        train_indices = indices[:split_idx]
        test_indices = indices[split_idx:]
        
        # Create datasets
        train_dataset = FakeNewsDataset()
        test_dataset = FakeNewsDataset()
        
        for idx in train_indices:
            train_dataset.add_article(self.articles[idx])
        
        for idx in test_indices:
            test_dataset.add_article(self.articles[idx])
        
        return train_dataset, test_dataset
    
    def get_texts(self) -> List[str]:
        """Get all article texts."""
        return [article.text for article in self.articles]
    
    def get_labels(self) -> np.ndarray:
        """Get all labels as numpy array."""
        return np.array([article.label for article in self.articles])
    
    def get_titles(self) -> List[str]:
        """Get all article titles."""
        return [article.title for article in self.articles]
    
    def __len__(self) -> int:
        """Return number of articles."""
        return len(self.articles)
    
    def __getitem__(self, idx: int) -> NewsArticle:
        """Get article by index."""
        return self.articles[idx]
    
    def __iter__(self):
        """Iterate over articles."""
        return iter(self.articles)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get dataset statistics.
        
        Returns:
            Dictionary with statistics
        """
        texts = self.get_texts()
        text_lengths = [len(text.split()) for text in texts]
        
        return {
            'num_articles': len(self.articles),
            'label_distribution': self.label_counts,
            'avg_text_length': np.mean(text_lengths),
            'median_text_length': np.median(text_lengths),
            'min_text_length': np.min(text_lengths),
            'max_text_length': np.max(text_lengths),
        }
    
    def save_to_json(self, filepath: str):
        """
        Save dataset to JSON file.
        
        Args:
            filepath: Output file path
        """
        data = [article.to_dict() for article in self.articles]
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(self.articles)} articles to {filepath}")
