"""
MappingNet for Video Prediction Reconstruction Compression.

This module contains two networks:
1. ConditionFusionNet: Fuses F_a and F_b features using gated fusion mechanism,
   outputs two low-rank decomposition vectors (U: 77*r, V: r*1024) for text embedding
2. NullTextNet: Processes F_a to generate null-text embedding (77*1024) for uncondition guidance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class GatedFusionModule(nn.Module):
    """
    Gated Fusion Module for combining F_a and F_b features.
    Uses attention-based gating to learn complementary information from both features.
    """
    
    def __init__(self, feature_dim: int, hidden_dim: int = 1024):
        """
        Args:
            feature_dim: Dimension of input features
            hidden_dim: Hidden dimension for fusion layers
        """
        super().__init__()
        
        # Feature projection layers
        self.proj_a = nn.Linear(feature_dim, hidden_dim)
        self.proj_b = nn.Linear(feature_dim, hidden_dim)
        
        # Gate computation
        self.gate_a = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        self.gate_b = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        # Cross attention for feature interaction
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Output fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, f_a: torch.Tensor, f_b: torch.Tensor) -> torch.Tensor:
        """
        Fuse F_a and F_b features using gated mechanism.
        
        Args:
            f_a: Features from image A (batch, num_tokens, feature_dim) or (batch, feature_dim)
            f_b: Features from image B (batch, num_tokens, feature_dim) or (batch, feature_dim)
            
        Returns:
            Fused features (batch, hidden_dim)
        """
        # Handle different input shapes
        if f_a.dim() == 3:
            # Use CLS token (first token) or mean pooling
            f_a_pooled = f_a[:, 0, :]  # CLS token
            f_b_pooled = f_b[:, 0, :]
        else:
            f_a_pooled = f_a
            f_b_pooled = f_b
            
        # Project features
        h_a = self.proj_a(f_a_pooled)  # (batch, hidden_dim)
        h_b = self.proj_b(f_b_pooled)  # (batch, hidden_dim)
        
        # Concatenate for gate computation
        h_concat = torch.cat([h_a, h_b], dim=-1)  # (batch, hidden_dim * 2)
        
        # Compute gates
        g_a = self.gate_a(h_concat)  # (batch, hidden_dim)
        g_b = self.gate_b(h_concat)  # (batch, hidden_dim)
        
        # Apply gates
        gated_a = g_a * h_a
        gated_b = g_b * h_b
        
        # Cross attention if we have token-level features
        if f_a.dim() == 3:
            proj_tokens_a = self.proj_a(f_a)  # (batch, num_tokens, hidden_dim)
            proj_tokens_b = self.proj_b(f_b)
            
            # B attends to A (learn what to take from reference)
            attn_out, _ = self.cross_attn(
                proj_tokens_b, proj_tokens_a, proj_tokens_a
            )
            attn_pooled = attn_out[:, 0, :]  # Use CLS position
            
            # Combine with gated features
            fused = self.fusion_layer(torch.cat([gated_a + gated_b, attn_pooled], dim=-1))
        else:
            fused = self.fusion_layer(torch.cat([gated_a, gated_b], dim=-1))
            
        return fused


class ConditionFusionNet(nn.Module):
    """
    Network for fusing F_a and F_b to generate low-rank text embedding condition.
    Outputs two vectors U (77, r) and V (r, 1024) for low-rank decomposition.
    """
    
    def __init__(
        self, 
        feature_dim: int = 1024,
        hidden_dim: int = 1024,
        seq_len: int = 77,
        embed_dim: int = 1024,
        rank: int = 8
    ):
        """
        Args:
            feature_dim: Input feature dimension from image encoder
            hidden_dim: Hidden dimension for fusion layers
            seq_len: Sequence length for text embedding (77 for SD)
            embed_dim: Text embedding dimension (1024 for SD-Turbo)
            rank: Rank for low-rank decomposition (controls bitrate)
        """
        super().__init__()
        
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.rank = rank
        # Cache sqrt(rank) for efficiency
        self.rank_sqrt = rank ** 0.5
        
        # Gated fusion module
        self.gated_fusion = GatedFusionModule(feature_dim, hidden_dim)
        
        # MLP to generate U matrix (77 * r)
        self.u_generator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, seq_len * rank)
        )
        
        # MLP to generate V matrix (r * 1024)
        self.v_generator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, rank * embed_dim)
        )
        
        # Initialize weights for stable training
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights with small values for stable training."""
        for module in [self.u_generator, self.v_generator]:
            if isinstance(module, nn.Sequential):
                for m in module:
                    if isinstance(m, nn.Linear):
                        nn.init.xavier_uniform_(m.weight, gain=0.1)
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)
    
    def forward(
        self, 
        f_a: torch.Tensor, 
        f_b: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate low-rank text embedding condition from fused features.
        
        Args:
            f_a: Features from image A
            f_b: Features from image B
            
        Returns:
            Tuple of:
                - U: Low-rank factor (batch, 77, r)
                - V: Low-rank factor (batch, r, 1024)
                - condition: Full condition matrix (batch, 77, 1024)
        """
        # Fuse features
        fused = self.gated_fusion(f_a, f_b)  # (batch, hidden_dim)
        
        # Generate U and V matrices
        u_flat = self.u_generator(fused)  # (batch, 77 * r)
        v_flat = self.v_generator(fused)  # (batch, r * 1024)
        
        # Reshape
        batch_size = fused.shape[0]
        U = u_flat.view(batch_size, self.seq_len, self.rank)  # (batch, 77, r)
        V = v_flat.view(batch_size, self.rank, self.embed_dim)  # (batch, r, 1024)
        
        # Compute full condition matrix via matrix multiplication
        # Scale by sqrt(rank) for stable training (similar to attention scaling)
        condition = torch.bmm(U, V) / self.rank_sqrt  # (batch, 77, 1024)
        
        return U, V, condition


class NullTextNet(nn.Module):
    """
    Network for generating null-text embedding from F_a.
    Used as unconditional guidance for classifier-free guidance.
    """
    
    def __init__(
        self,
        feature_dim: int = 1024,
        hidden_dim: int = 1024,
        seq_len: int = 77,
        embed_dim: int = 1024,
        num_layers: int = 4
    ):
        """
        Args:
            feature_dim: Input feature dimension from image encoder
            hidden_dim: Hidden dimension for MLP layers
            seq_len: Sequence length for null-text embedding (77 for SD)
            embed_dim: Embedding dimension (1024 for SD-Turbo)
            num_layers: Number of MLP layers
        """
        super().__init__()
        
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        
        # Feature pooling if input has tokens
        self.feature_pool = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # Build MLP layers
        layers = []
        for i in range(num_layers):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1)
            ])
        self.mlp = nn.Sequential(*layers)
        
        # Output projection to null-text embedding
        self.output_proj = nn.Linear(hidden_dim, seq_len * embed_dim)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights with small values."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, f_a: torch.Tensor) -> torch.Tensor:
        """
        Generate null-text embedding from F_a features.
        
        Args:
            f_a: Features from image A (batch, num_tokens, feature_dim) or (batch, feature_dim)
            
        Returns:
            null_text: Null-text embedding (batch, 77, 1024)
        """
        # Pool tokens if necessary
        if f_a.dim() == 3:
            # Use CLS token
            f_a = f_a[:, 0, :]
            
        # Project and transform
        h = self.feature_pool(f_a)  # (batch, hidden_dim)
        h = self.mlp(h)  # (batch, hidden_dim)
        
        # Generate null-text embedding
        batch_size = f_a.shape[0]
        null_text = self.output_proj(h)  # (batch, seq_len * embed_dim)
        null_text = null_text.view(batch_size, self.seq_len, self.embed_dim)
        
        return null_text


class MappingNet(nn.Module):
    """
    Complete MappingNet combining ConditionFusionNet and NullTextNet.
    
    Architecture:
    - ConditionFusionNet: Fuses F_a and F_b with gated mechanism, outputs U (77*r), V (r*1024)
    - NullTextNet: Processes F_a to output null-text embedding (77*1024)
    """
    
    def __init__(
        self,
        feature_dim: int = 1024,
        hidden_dim: int = 1024,
        seq_len: int = 77,
        embed_dim: int = 1024,
        rank: int = 8
    ):
        """
        Args:
            feature_dim: Input feature dimension from image encoder
            hidden_dim: Hidden dimension for networks
            seq_len: Sequence length for embeddings (77 for SD)
            embed_dim: Embedding dimension (1024 for SD-Turbo)
            rank: Rank for low-rank decomposition
        """
        super().__init__()
        
        self.rank = rank
        
        # Condition fusion network for text embedding
        self.condition_net = ConditionFusionNet(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            seq_len=seq_len,
            embed_dim=embed_dim,
            rank=rank
        )
        
        # Null-text embedding network
        self.null_text_net = NullTextNet(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            seq_len=seq_len,
            embed_dim=embed_dim
        )
        
    def forward(
        self, 
        f_a: torch.Tensor, 
        f_b: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate both conditional and unconditional embeddings.
        
        Args:
            f_a: Features from image A (reference frame)
            f_b: Features from image B (target frame)
            
        Returns:
            Tuple of:
                - U: Low-rank factor (batch, 77, r)
                - V: Low-rank factor (batch, r, 1024)
                - condition: Full condition matrix (batch, 77, 1024)
                - null_text: Null-text embedding (batch, 77, 1024)
        """
        # Generate text embedding condition
        U, V, condition = self.condition_net(f_a, f_b)
        
        # Generate null-text embedding
        null_text = self.null_text_net(f_a)
        
        return U, V, condition, null_text
    
    def get_rank(self) -> int:
        """Return the rank used for low-rank decomposition."""
        return self.rank
