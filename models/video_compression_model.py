"""
Video Compression Model for Video Prediction Reconstruction.

This module combines:
1. Image Encoder (CLIP ViT backbone)
2. MappingNet (Gated Fusion + Null-Text networks)
3. SD-Turbo (Diffusion model for reconstruction)
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional

from .image_encoder import CLIPImageEncoder
from .mapping_net import MappingNet


class VideoCompressionModel(nn.Module):
    """
    Complete Video Compression Model for Video Prediction Reconstruction.
    
    Architecture:
    - Image Encoder: CLIP ViT backbone to encode images A and B
    - MappingNet: 
        - ConditionFusionNet: Fuses F_a and F_b → low-rank condition (U, V)
        - NullTextNet: F_a → null-text embedding
    - SD-Turbo: Diffusion model (external, passed during training)
    
    Training:
    - VAE encodes image B to latent space
    - Add noise for warm start
    - Run diffusion with MappingNet conditions
    - Supervise with MSE loss in latent space + regularization
    """
    
    def __init__(
        self,
        clip_arch: str = "ViT-L-14",
        clip_version: str = "openai",
        hidden_dim: int = 1024,
        seq_len: int = 77,
        embed_dim: int = 1024,
        rank: int = 8,
        freeze_encoder: bool = True,
        device: str = "cuda"
    ):
        """
        Args:
            clip_arch: CLIP architecture (ViT-L-14 for rich features)
            clip_version: Pretrained CLIP version
            hidden_dim: Hidden dimension for MappingNet
            seq_len: Sequence length for embeddings (77 for SD)
            embed_dim: Embedding dimension (1024 for SD-Turbo)
            rank: Rank for low-rank decomposition (controls bitrate)
            freeze_encoder: Whether to freeze the image encoder
            device: Device to use
        """
        super().__init__()
        
        self.device = device
        self.rank = rank
        
        # Image Encoder using CLIP ViT
        self.image_encoder = CLIPImageEncoder(
            arch=clip_arch,
            version=clip_version,
            device=device,
            freeze=freeze_encoder,
        )
        
        # Get feature dimension from encoder
        feature_dim = self.image_encoder.get_feature_dim()
        
        # MappingNet for condition generation
        self.mapping_net = MappingNet(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            seq_len=seq_len,
            embed_dim=embed_dim,
            rank=rank
        )
        
    def encode_images(
        self, 
        img_a: torch.Tensor, 
        img_b: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode image pair using CLIP encoder.
        
        Args:
            img_a: Reference image (batch, 3, H, W) in range [-1, 1]
            img_b: Target image (batch, 3, H, W) in range [-1, 1]
            
        Returns:
            Tuple of (F_a, F_b) feature tensors
        """
        F_a, F_b = self.image_encoder.encode_pair(img_a, img_b)
        return F_a, F_b
    
    def generate_conditions(
        self,
        F_a: torch.Tensor,
        F_b: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Generate conditioning tensors from image features.
        
        Args:
            F_a: Features from image A
            F_b: Features from image B
            
        Returns:
            Dictionary containing:
                - 'U': Low-rank factor U (batch, 77, r)
                - 'V': Low-rank factor V (batch, r, 1024)
                - 'condition': Full condition (batch, 77, 1024)
                - 'null_text': Null-text embedding (batch, 77, 1024)
        """
        U, V, condition, null_text = self.mapping_net(F_a, F_b)
        
        return {
            'U': U,
            'V': V,
            'condition': condition,
            'null_text': null_text
        }
    
    def forward(
        self,
        img_a: torch.Tensor,
        img_b: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass to generate all conditioning tensors.
        
        Args:
            img_a: Reference image (previous frame)
            img_b: Target image (current frame)
            
        Returns:
            Dictionary with condition tensors and features
        """
        # Encode images
        F_a, F_b = self.encode_images(img_a, img_b)
        
        # Generate conditions
        conditions = self.generate_conditions(F_a, F_b)
        
        # Add features to output
        conditions['F_a'] = F_a
        conditions['F_b'] = F_b
        
        return conditions
    
    def get_trainable_parameters(self):
        """Get parameters that should be trained (MappingNet only)."""
        return self.mapping_net.parameters()
    
    def get_rank(self) -> int:
        """Return the rank for low-rank decomposition."""
        return self.rank
    
    def compute_condition_from_uv(
        self, 
        U: torch.Tensor, 
        V: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute full condition matrix from U and V factors.
        
        Args:
            U: Low-rank factor (batch, 77, r)
            V: Low-rank factor (batch, r, 1024)
            
        Returns:
            condition: Full condition matrix (batch, 77, 1024)
        """
        return torch.bmm(U, V) / (self.rank ** 0.5)
