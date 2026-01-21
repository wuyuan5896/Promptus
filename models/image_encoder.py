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
    - CLS token provides a global image representation (1024-dim for ViT-L/14)
    
    Feature dimensions:
    - For ViT-L/14: 1024-dimensional features
    - Output shape: (batch, 1024) - CLS token representation
    """
    
    def __init__(
        self,
        arch: str = "ViT-L-14",
        version: str = "openai",
        device: str = "cuda",
        freeze: bool = True,
    ):
        """
        Args:
            arch: Architecture of the CLIP model (ViT-L-14 for rich features)
            version: Pretrained weights version
            device: Device to run on
            freeze: Whether to freeze the encoder weights
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
        
        # Delete text components to save memory
        if hasattr(model, 'transformer'):
            del model.transformer
        if hasattr(model, 'token_embedding'):
            del model.token_embedding
        if hasattr(model, 'positional_embedding'):
            del model.positional_embedding
        if hasattr(model, 'ln_final'):
            del model.ln_final
        if hasattr(model, 'text_projection'):
            del model.text_projection
        
        self.device = device
        
        # Get the feature dimension from the model
        if hasattr(self.visual, 'output_dim'):
            self.feature_dim = self.visual.output_dim
        elif hasattr(self.visual, 'embed_dim'):
            self.feature_dim = self.visual.embed_dim
        else:
            # Default for ViT-L-14
            self.feature_dim = 1024
        
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
            Feature tensor of shape (batch, feature_dim) - CLS token representation
        """
        x = self.preprocess(x)
        
        with torch.no_grad() if not self.training else torch.enable_grad():
            # Get visual features (CLS token)
            features = self.visual(x)
            
            # Handle case where output might be a tuple
            if isinstance(features, tuple):
                features = features[0]
            
            # Ensure 2D output (batch, feature_dim)
            if features.dim() == 3:
                # Take CLS token (first token)
                features = features[:, 0, :]
            
            return features
    
    def encode_pair(self, img_a: torch.Tensor, img_b: torch.Tensor):
        """
        Encode a pair of images.
        
        Args:
            img_a: Reference image (previous frame) in range [-1, 1]
            img_b: Target image (current frame) in range [-1, 1]
            
        Returns:
            Tuple of (F_a, F_b) feature tensors, each (batch, feature_dim)
        """
        F_a = self.forward(img_a)
        F_b = self.forward(img_b)
        return F_a, F_b
    
    def get_feature_dim(self) -> int:
        """Return the feature dimension of the encoder."""
        return self.feature_dim
