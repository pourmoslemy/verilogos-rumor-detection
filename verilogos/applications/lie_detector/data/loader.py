"""
Data Loader Module

Utilities for loading and managing fake news datasets.

Author: Alireza Pourmoslemi
Email: apmath99@gmail.com
"""

import json
import os
from typing import List, Dict, Any, Optional
from pathlib import Path
import requests


class DataLoader:
    """
    Utility class for loading fake news datasets.
    
    Supports:
        - Local JSON files
        - FakeNewsNet format
        - Custom formats
    """
    
    def __init__(self, cache_dir: str = "./data/cache"):
        """
        Initialize data loader.
        
        Args:
            cache_dir: Directory for caching downloaded data
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def load_json(self, filepath: str) -> List[Dict[str, Any]]:
        """
        Load data from JSON file.
        
        Args:
            filepath: Path to JSON file
            
        Returns:
            List of article dictionaries
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, dict):
            # Handle nested format
            if 'articles' in data:
                return data['articles']
            return [data]
        
        return data
    
    def load_fakenewsnet(self, data_dir: str) -> List[Dict[str, Any]]:
        """
        Load FakeNewsNet dataset.
        
        Expected structure:
            data_dir/
                fake/
                    article1.json
                    article2.json
                real/
                    article3.json
                    article4.json
        
        Args:
            data_dir: Root directory of FakeNewsNet data
            
        Returns:
            List of article dictionaries with labels
        """
        data_dir = Path(data_dir)
        articles = []
        
        # Load fake articles
        fake_dir = data_dir / 'fake'
        if fake_dir.exists():
            for json_file in fake_dir.glob('*.json'):
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        article = json.load(f)
                    article['label'] = 0  # Fake
                    article['id'] = json_file.stem
                    articles.append(article)
                except Exception as e:
                    print(f"Error loading {json_file}: {e}")
        
        # Load real articles
        real_dir = data_dir / 'real'
        if real_dir.exists():
            for json_file in real_dir.glob('*.json'):
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        article = json.load(f)
                    article['label'] = 1  # Real
                    article['id'] = json_file.stem
                    articles.append(article)
                except Exception as e:
                    print(f"Error loading {json_file}: {e}")
        
        return articles
    
    def create_sample_dataset(self, output_path: str, num_samples: int = 100):
        """
        Create sample dataset for testing.
        
        Args:
            output_path: Path to save sample dataset
            num_samples: Number of samples to generate
        """
        import random
        
        fake_templates = [
            "Breaking: Scientists discover {topic}. Experts shocked!",
            "You won't believe what {person} said about {topic}!",
            "URGENT: {topic} causes {effect}. Share immediately!",
            "Secret revealed: {topic} linked to {conspiracy}.",
        ]
        
        real_templates = [
            "Study shows {topic} may affect {outcome}.",
            "Researchers investigate {topic} in new study.",
            "Analysis: {topic} trends over past decade.",
            "Report: {topic} impacts {area} according to data.",
        ]
        
        topics = ["climate change", "vaccines", "economy", "technology", "health"]
        persons = ["celebrity", "politician", "scientist", "expert"]
        effects = ["cancer", "disease", "problems", "issues"]
        conspiracies = ["government plot", "corporate scheme", "hidden agenda"]
        outcomes = ["public health", "environment", "society"]
        areas = ["healthcare", "education", "industry"]
        
        articles = []
        
        for i in range(num_samples):
            if i % 2 == 0:
                # Fake article
                template = random.choice(fake_templates)
                text = template.format(
                    topic=random.choice(topics),
                    person=random.choice(persons),
                    effect=random.choice(effects),
                    conspiracy=random.choice(conspiracies),
                )
                label = 0
            else:
                # Real article
                template = random.choice(real_templates)
                text = template.format(
                    topic=random.choice(topics),
                    outcome=random.choice(outcomes),
                    area=random.choice(areas),
                )
                label = 1
            
            articles.append({
                'id': f'sample_{i}',
                'title': text,
                'text': text + " " + " ".join([random.choice(topics) for _ in range(20)]),
                'label': label,
                'source': 'sample',
                'date': '2025-01-01',
            })
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(articles, f, indent=2)
        
        print(f"Created sample dataset with {num_samples} articles at {output_path}")
