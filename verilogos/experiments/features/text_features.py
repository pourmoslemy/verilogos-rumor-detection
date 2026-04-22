"""
Text Features - Extract text-based features

Computes TF-IDF vectors and optional embeddings.

Author: Alireza Pourmoslemi
Email: apmath99@gmail.com
"""

import numpy as np
from typing import List, Dict, Any, Optional

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available for text features")


class TextFeatureExtractor:
    """
    Extract text features using TF-IDF.
    
    Example:
        >>> extractor = TextFeatureExtractor(max_features=1000)
        >>> X_train = extractor.fit_transform(train_texts)
        >>> X_test = extractor.transform(test_texts)
    """
    
    def __init__(
        self,
        max_features: int = 1000,
        ngram_range: tuple = (1, 2),
        min_df: int = 2,
        max_df: float = 0.95
    ):
        """
        Initialize text feature extractor.
        
        Args:
            max_features: Maximum number of features
            ngram_range: N-gram range (min, max)
            min_df: Minimum document frequency
            max_df: Maximum document frequency
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required for text features")
        
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            stop_words='english'
        )
        
        self.is_fitted = False
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """
        Fit vectorizer and transform texts.
        
        Args:
            texts: List of texts
        
        Returns:
            Feature matrix (n_samples, n_features)
        """
        X = self.vectorizer.fit_transform(texts)
        self.is_fitted = True
        
        print(f"Extracted {X.shape[1]} text features from {X.shape[0]} samples")
        
        return X.toarray()
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """
        Transform texts using fitted vectorizer.
        
        Args:
            texts: List of texts
        
        Returns:
            Feature matrix
        """
        if not self.is_fitted:
            raise ValueError("Vectorizer not fitted. Call fit_transform() first.")
        
        X = self.vectorizer.transform(texts)
        return X.toarray()
    
    def get_feature_names(self) -> List[str]:
        """Get feature names."""
        if not self.is_fitted:
            raise ValueError("Vectorizer not fitted")
        
        return self.vectorizer.get_feature_names_out().tolist()


def extract_text_features(
    texts: List[str],
    method: str = 'tfidf',
    max_features: int = 1000,
    **kwargs
) -> np.ndarray:
    """
    Extract text features from list of texts.
    
    Args:
        texts: List of texts
        method: Feature extraction method ('tfidf')
        max_features: Maximum number of features
        **kwargs: Additional parameters
    
    Returns:
        Feature matrix (n_samples, n_features)
    """
    if method == 'tfidf':
        extractor = TextFeatureExtractor(max_features=max_features, **kwargs)
        return extractor.fit_transform(texts)
    else:
        raise ValueError(f"Unknown method: {method}")


def extract_basic_text_stats(text: str) -> Dict[str, float]:
    """
    Extract basic text statistics.
    
    Args:
        text: Input text
    
    Returns:
        Dictionary of statistics
    """
    words = text.split()
    sentences = text.split('.')
    
    stats = {
        'num_words': len(words),
        'num_sentences': len(sentences),
        'avg_word_length': np.mean([len(w) for w in words]) if words else 0,
        'num_chars': len(text),
        'num_unique_words': len(set(words)),
        'lexical_diversity': len(set(words)) / len(words) if words else 0
    }
    
    return stats


def extract_batch_text_stats(texts: List[str]) -> np.ndarray:
    """
    Extract basic text statistics for multiple texts.
    
    Args:
        texts: List of texts
    
    Returns:
        Feature matrix (n_samples, n_features)
    """
    all_stats = []
    
    for text in texts:
        stats = extract_basic_text_stats(text)
        all_stats.append(stats)
    
    # Convert to matrix
    if all_stats:
        feature_names = sorted(all_stats[0].keys())
        X = np.zeros((len(all_stats), len(feature_names)))
        
        for i, stats in enumerate(all_stats):
            X[i] = [stats[name] for name in feature_names]
        
        return X, feature_names
    else:
        return np.array([]), []


class CombinedFeatureExtractor:
    """
    Combine topology and text features.
    
    Example:
        >>> extractor = CombinedFeatureExtractor()
        >>> X_train = extractor.fit_transform(train_texts)
        >>> X_test = extractor.transform(test_texts)
    """
    
    def __init__(
        self,
        use_topology: bool = True,
        use_text: bool = True,
        max_text_features: int = 500,
        similarity_threshold: float = 0.3
    ):
        """
        Initialize combined extractor.
        
        Args:
            use_topology: Whether to use topology features
            use_text: Whether to use text features
            max_text_features: Max text features
            similarity_threshold: Topology threshold
        """
        self.use_topology = use_topology
        self.use_text = use_text
        self.max_text_features = max_text_features
        self.similarity_threshold = similarity_threshold
        
        if use_text:
            self.text_extractor = TextFeatureExtractor(
                max_features=max_text_features
            )
        
        self.is_fitted = False
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """
        Fit and transform texts.
        
        Args:
            texts: List of texts
        
        Returns:
            Combined feature matrix
        """
        features = []
        
        # Extract topology features
        if self.use_topology:
            from verilogos.experiments.features.topology_features import extract_batch_topology_features
            
            print("Extracting topology features...")
            X_topo, _ = extract_batch_topology_features(
                texts,
                similarity_threshold=self.similarity_threshold
            )
            features.append(X_topo)
        
        # Extract text features
        if self.use_text:
            print("Extracting text features...")
            X_text = self.text_extractor.fit_transform(texts)
            features.append(X_text)
        
        # Combine
        if features:
            X_combined = np.hstack(features)
            self.is_fitted = True
            
            print(f"Combined features: {X_combined.shape}")
            return X_combined
        else:
            raise ValueError("No features selected")
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """
        Transform texts using fitted extractors.
        
        Args:
            texts: List of texts
        
        Returns:
            Combined feature matrix
        """
        if not self.is_fitted:
            raise ValueError("Extractor not fitted")
        
        features = []
        
        if self.use_topology:
            from verilogos.experiments.features.topology_features import extract_batch_topology_features
            
            X_topo, _ = extract_batch_topology_features(
                texts,
                similarity_threshold=self.similarity_threshold,
                show_progress=False
            )
            features.append(X_topo)
        
        if self.use_text:
            X_text = self.text_extractor.transform(texts)
            features.append(X_text)
        
        return np.hstack(features)
