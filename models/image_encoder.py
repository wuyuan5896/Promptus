"""
Image Encoder using CLIP ViT backbone.
Encodes images A and B into feature representations F_a and F_b.
Uses OpenCLIP's ViT-L/14 which provides rich semantic and detail features.
"""

import torch
import torch.nn as nn
import open_clip
import kornia


class CLIPImageEncoder(nn.Module):
    """
    CLIP Image Encoder using ViT backbone.
    
    Uses OpenCLIP's ViT-L/14 architecture which provides:
    - Rich semantic features from the vision transformer
    - Output tokens with shape (batch, 257, 1024) where 257 = 1 CLS + 256 patch tokens
    - The CLS token provides a global image representation
    
    Feature dimensions:
    - For ViT-L/14: 1024-dimensional features
    - Output shape: (batch, 257, 1024) when output_tokens=True
    - Or (batch, 1024) for CLS token only
    """
    
    def __init__(
        self,
        arch: str = "ViT-L-14",
        version: str = "openai",
        device: str = "cuda",
        freeze: bool = True,
        output_tokens: bool = True,
    ):
        """
        Args:
            arch: Architecture of the CLIP model (ViT-L-14 for rich features)
            version: Pretrained weights version
            device: Device to run on
            freeze: Whether to freeze the encoder weights
            output_tokens: Whether to output all tokens or just CLS token
        """
        super().__init__()
        
        # Create CLIP model
        model, _, preprocess = open_clip.create_model_and_transforms(
            arch,
            device=torch.device("cpu"),
            pretrained=version,
        )
        
        # Only keep the visual encoder
        self.visual = model.visual
        del model.transformer
        del model.token_embedding
        del model.positional_embedding
        del model.ln_final
        del model.text_projection
        
        self.device = device
        self.output_tokens = output_tokens
        
        # Get the feature dimension from the model
        self.feature_dim = self.visual.output_dim  # 1024 for ViT-L-14
        
        # Register normalization buffers
        self.register_buffer(
            "mean", 
            torch.Tensor([0.48145466, 0.4578275, 0.40821073]), 
            persistent=False
        )
        self.register_buffer(
            "std", 
            torch.Tensor([0.26862954, 0.26130258, 0.27577711]), 
            persistent=False
        )
        
        if freeze:
            self.freeze()
            
        # Enable output tokens if needed
        if hasattr(self.visual, 'output_tokens'):
            self.visual.output_tokens = output_tokens
    
    def freeze(self):
        """Freeze all encoder parameters."""
        self.visual.eval()
        for param in self.visual.parameters():
            param.requires_grad = False
    
    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """
        Preprocess images for CLIP.
        
        Args:
            x: Input tensor of shape (batch, 3, H, W) in range [-1, 1]
            
        Returns:
            Preprocessed tensor of shape (batch, 3, 224, 224)
        """
        # Resize to 224x224
        x = kornia.geometry.resize(
            x,
            (224, 224),
            interpolation="bicubic",
            align_corners=True,
            antialias=True,
        )
        # Normalize from [-1, 1] to [0, 1]
        x = (x + 1.0) / 2.0
        # Apply CLIP normalization
        x = kornia.enhance.normalize(x, self.mean, self.std)
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode image to feature representation.
        
        Args:
            x: Input image tensor of shape (batch, 3, H, W) in range [-1, 1]
            
        Returns:
            If output_tokens=True: (batch, num_tokens+1, feature_dim) 
                - includes CLS token + patch tokens
            If output_tokens=False: (batch, feature_dim) 
                - CLS token only
        """
        x = self.preprocess(x)
        
        with torch.no_grad() if not self.training else torch.enable_grad():
            if self.output_tokens and hasattr(self.visual, 'output_tokens') and self.visual.output_tokens:
                cls_token, patch_tokens = self.visual(x)
                # Combine CLS token with patch tokens
                # cls_token: (batch, feature_dim)
                # patch_tokens: (batch, num_patches, feature_dim)
                features = torch.cat([cls_token.unsqueeze(1), patch_tokens], dim=1)
                return features
            else:
                # Just return CLS token
                features = self.visual(x)
                return features
    
    def encode_pair(self, img_a: torch.Tensor, img_b: torch.Tensor):
        """
        Encode a pair of images.
        
        Args:
            img_a: Reference image (previous frame) in range [-1, 1]
            img_b: Target image (current frame) in range [-1, 1]
            
        Returns:
            Tuple of (F_a, F_b) feature tensors
        """
        F_a = self.forward(img_a)
        F_b = self.forward(img_b)
        return F_a, F_b
    
    def get_feature_dim(self) -> int:
        """Return the feature dimension of the encoder."""
        return self.feature_dim
    
    def get_num_tokens(self) -> int:
        """Return the number of tokens (including CLS) for ViT-L/14 at 224x224."""
        # 224x224 image with patch size 14 -> 16x16 = 256 patches + 1 CLS = 257
        return 257
