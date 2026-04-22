"""
Text Preprocessor Module

Preprocessing pipeline for text data before topological analysis.

Author: Alireza Pourmoslemi
Email: apmath99@gmail.com
"""

import re
from typing import List, Dict, Set
import string


class TextPreprocessor:
    """
    Text preprocessing for fake news detection.
    
    Performs:
        - Sentence segmentation
        - Tokenization
        - Stopword removal
        - Lowercasing
        - Punctuation handling
    
    Example:
        >>> preprocessor = TextPreprocessor()
        >>> sentences = preprocessor.segment_sentences(text)
        >>> tokens = preprocessor.tokenize(text)
    """
    
    def __init__(self, remove_stopwords: bool = True):
        """
        Initialize preprocessor.
        
        Args:
            remove_stopwords: Whether to remove stopwords
        """
        self.remove_stopwords = remove_stopwords
        self.stopwords = self._load_stopwords()
    
    def _load_stopwords(self) -> Set[str]:
        """Load common English stopwords."""
        return {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'this', 'but', 'they', 'have', 'had',
            'what', 'when', 'where', 'who', 'which', 'why', 'how', 'all', 'each',
            'every', 'both', 'few', 'more', 'most', 'other', 'some', 'such',
        }
    
    def segment_sentences(self, text: str) -> List[str]:
        """
        Segment text into sentences.
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        # Simple sentence segmentation
        # Split on period, exclamation, question mark followed by space
        sentences = re.split(r'[.!?]+\s+', text)
        
        # Clean and filter
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        # Lowercase
        text = text.lower()
        
        # Remove punctuation except apostrophes
        text = re.sub(r"[^\w\s']", ' ', text)
        
        # Split on whitespace
        tokens = text.split()
        
        # Remove stopwords if enabled
        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in self.stopwords]
        
        # Filter empty and very short tokens
        tokens = [t for t in tokens if len(t) > 1]
        
        return tokens
    
    def preprocess(self, text: str) -> Dict[str, any]:
        """
        Complete preprocessing pipeline.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with preprocessed components
        """
        sentences = self.segment_sentences(text)
        tokens = self.tokenize(text)
        
        # Tokenize each sentence
        sentence_tokens = [self.tokenize(sent) for sent in sentences]
        
        return {
            'original': text,
            'sentences': sentences,
            'tokens': tokens,
            'sentence_tokens': sentence_tokens,
            'num_sentences': len(sentences),
            'num_tokens': len(tokens),
        }
    
    def extract_entities(self, text: str) -> List[str]:
        """
        Extract named entities (simple version).
        
        Args:
            text: Input text
            
        Returns:
            List of potential entities (capitalized words)
        """
        # Simple heuristic: capitalized words that aren't sentence starts
        words = text.split()
        entities = []
        
        for i, word in enumerate(words):
            # Skip first word of sentences
            if i > 0 and word[0].isupper():
                # Remove punctuation
                entity = re.sub(r'[^\w\s]', '', word)
                if len(entity) > 1:
                    entities.append(entity)
        
        return list(set(entities))
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute simple Jaccard similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score in [0, 1]
        """
        tokens1 = set(self.tokenize(text1))
        tokens2 = set(self.tokenize(text2))
        
        if not tokens1 or not tokens2:
            return 0.0
        
        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)
        
        return len(intersection) / len(union) if union else 0.0
