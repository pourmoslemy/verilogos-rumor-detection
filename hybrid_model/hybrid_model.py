"""
Hybrid Neural Network Architecture for Fake News Detection
Combines TDA (Topological) and Text (Semantic) modalities with Cross-Attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from typing import Dict, Optional, Tuple
import math


class TDAProjectionNetwork(nn.Module):
    """
    Projects 36-dimensional TDA features into 768-dimensional space.
    Uses MLP with BatchNorm and Dropout for robust representation learning.
    """
    
    def __init__(
        self,
        input_dim: int = 36,
        hidden_dims: list = [128, 256, 512],
        output_dim: int = 768,
        dropout: float = 0.3
    ):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Final projection to match text embedding dimension
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, tda_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tda_features: (batch_size, 36)
        
        Returns:
            projected_features: (batch_size, 768)
        """
        return self.network(tda_features)


class MultiHeadCrossAttention(nn.Module):
    """
    Multi-Head Cross-Attention mechanism for fusing TDA and Text modalities.
    
    Implements bidirectional attention:
    - Text attends to TDA (Text as Query, TDA as Key/Value)
    - TDA attends to Text (TDA as Query, Text as Key/Value)
    """
    
    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = math.sqrt(self.head_dim)
        
        # Projection layers for Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            query: (batch_size, embed_dim)
            key: (batch_size, embed_dim)
            value: (batch_size, embed_dim)
        
        Returns:
            attended_output: (batch_size, embed_dim)
        """
        batch_size = query.size(0)
        
        # Project and reshape for multi-head attention
        Q = self.q_proj(query).view(batch_size, self.num_heads, self.head_dim)
        K = self.k_proj(key).view(batch_size, self.num_heads, self.head_dim)
        V = self.v_proj(value).view(batch_size, self.num_heads, self.head_dim)
        
        # Compute attention scores
        # (batch_size, num_heads, head_dim) @ (batch_size, num_heads, head_dim).T
        # -> (batch_size, num_heads, 1, 1) but we need scalar per head
        scores = torch.einsum('bhd,bhd->bh', Q, K) / self.scale  # (batch_size, num_heads)
        
        # Apply softmax (across heads dimension)
        attn_weights = F.softmax(scores, dim=-1)  # (batch_size, num_heads)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        # (batch_size, num_heads) * (batch_size, num_heads, head_dim)
        attended = torch.einsum('bh,bhd->bhd', attn_weights, V)  # (batch_size, num_heads, head_dim)
        
        # Concatenate heads
        attended = attended.contiguous().view(batch_size, self.embed_dim)
        
        # Final projection
        output = self.out_proj(attended)
        
        return output


class HybridFakeNewsDetector(nn.Module):
    """
    Hybrid Neural Network combining TDA and Text modalities.
    
    Architecture:
    1. Text Pathway: Pre-trained transformer (DistilRoBERTa/BERT) -> [CLS] embedding
    2. TDA Pathway: 36D features -> MLP projection -> 768D
    3. Fusion: Bidirectional Cross-Attention
    4. Classification: Fused representation -> Binary classifier
    
    Supports three modes:
    - 'tda_only': Uses only topological features
    - 'text_only': Uses only text features
    - 'hybrid': Uses both with cross-attention fusion
    """
    
    def __init__(
        self,
        mode: str = 'hybrid',
        text_model_name: str = 'distilroberta-base',
        tda_input_dim: int = 36,
        tda_hidden_dims: list = [128, 256, 512],
        embed_dim: int = 768,
        num_attention_heads: int = 8,
        dropout: float = 0.3,
        freeze_text_encoder: bool = False
    ):
        super().__init__()
        
        assert mode in ['tda_only', 'text_only', 'hybrid'], f"Invalid mode: {mode}"
        
        self.mode = mode
        self.embed_dim = embed_dim
        
        # Text encoder
        if mode in ['text_only', 'hybrid']:
            self.text_encoder = AutoModel.from_pretrained(text_model_name)
            
            if freeze_text_encoder:
                for param in self.text_encoder.parameters():
                    param.requires_grad = False
        else:
            self.text_encoder = None
        
        # TDA projection network
        if mode in ['tda_only', 'hybrid']:
            self.tda_projection = TDAProjectionNetwork(
                input_dim=tda_input_dim,
                hidden_dims=tda_hidden_dims,
                output_dim=embed_dim,
                dropout=dropout
            )
        else:
            self.tda_projection = None
        
        # Cross-attention for hybrid mode
        if mode == 'hybrid':
            self.text_to_tda_attention = MultiHeadCrossAttention(
                embed_dim=embed_dim,
                num_heads=num_attention_heads,
                dropout=dropout
            )
            self.tda_to_text_attention = MultiHeadCrossAttention(
                embed_dim=embed_dim,
                num_heads=num_attention_heads,
                dropout=dropout
            )
            
            # Fusion layer
            self.fusion_layer = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        
        # Classification head
        if mode == 'hybrid':
            classifier_input_dim = embed_dim
        else:
            classifier_input_dim = embed_dim
        
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 2)  # Binary classification
        )
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through the hybrid network.
        
        Args:
            batch: Dictionary containing:
                - 'tda_features': (batch_size, 36) if mode includes TDA
                - 'input_ids': (batch_size, seq_len) if mode includes text
                - 'attention_mask': (batch_size, seq_len) if mode includes text
        
        Returns:
            logits: (batch_size, 2) - class logits
        """
        if self.mode == 'tda_only':
            return self._forward_tda_only(batch)
        elif self.mode == 'text_only':
            return self._forward_text_only(batch)
        else:  # hybrid
            return self._forward_hybrid(batch)
    
    def _forward_tda_only(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """TDA-only forward pass."""
        tda_features = batch['tda_features']  # (batch_size, 36)
        
        # Project TDA features
        tda_embed = self.tda_projection(tda_features)  # (batch_size, 768)
        
        # Classify
        logits = self.classifier(tda_embed)  # (batch_size, 2)
        
        return logits
    
    def _forward_text_only(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Text-only forward pass."""
        input_ids = batch['input_ids']  # (batch_size, seq_len)
        attention_mask = batch['attention_mask']  # (batch_size, seq_len)
        
        # Encode text
        text_output = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Extract [CLS] token embedding
        text_embed = text_output.last_hidden_state[:, 0, :]  # (batch_size, 768)
        
        # Classify
        logits = self.classifier(text_embed)  # (batch_size, 2)
        
        return logits
    
    def _forward_hybrid(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Hybrid forward pass with cross-attention fusion."""
        # Extract TDA features
        tda_features = batch['tda_features']  # (batch_size, 36)
        tda_embed = self.tda_projection(tda_features)  # (batch_size, 768)
        
        # Extract text features
        input_ids = batch['input_ids']  # (batch_size, seq_len)
        attention_mask = batch['attention_mask']  # (batch_size, seq_len)
        
        text_output = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        text_embed = text_output.last_hidden_state[:, 0, :]  # (batch_size, 768)
        
        # Bidirectional cross-attention
        # Text attends to TDA
        text_attended = self.text_to_tda_attention(
            query=text_embed,
            key=tda_embed,
            value=tda_embed
        )  # (batch_size, 768)
        
        # TDA attends to Text
        tda_attended = self.tda_to_text_attention(
            query=tda_embed,
            key=text_embed,
            value=text_embed
        )  # (batch_size, 768)
        
        # Concatenate attended representations
        fused = torch.cat([text_attended, tda_attended], dim=-1)  # (batch_size, 1536)
        
        # Fusion layer
        fused_embed = self.fusion_layer(fused)  # (batch_size, 768)
        
        # Classify
        logits = self.classifier(fused_embed)  # (batch_size, 2)
        
        return logits
    
    def get_attention_weights(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Extract attention weights for interpretability.
        Only works in hybrid mode.
        
        Returns:
            {
                'text_to_tda_weights': (batch_size, num_heads),
                'tda_to_text_weights': (batch_size, num_heads)
            }
        """
        if self.mode != 'hybrid':
            raise ValueError("Attention weights only available in hybrid mode")
        
        # This is a simplified version - full implementation would require
        # modifying MultiHeadCrossAttention to return attention weights
        return {}


def create_model(
    mode: str,
    text_model_name: str = 'distilroberta-base',
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    **kwargs
) -> HybridFakeNewsDetector:
    """
    Factory function to create and initialize model.
    
    Args:
        mode: 'tda_only', 'text_only', or 'hybrid'
        text_model_name: HuggingFace model name
        device: Device to place model on
        **kwargs: Additional arguments for HybridFakeNewsDetector
    
    Returns:
        Initialized model on specified device
    """
    model = HybridFakeNewsDetector(
        mode=mode,
        text_model_name=text_model_name,
        **kwargs
    )
    
    model = model.to(device)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel: {mode.upper()}")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Device: {device}")
    
    return model
