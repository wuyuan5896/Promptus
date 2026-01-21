"""
Inference script for Video Prediction Reconstruction Compression.

Uses trained MappingNet to generate conditions for reconstructing video frames.
"""

import os
import argparse
import torch
import numpy as np
import cv2
from tqdm import tqdm

from diffusers import AutoencoderTiny
from scripts.demo.streamlit_helpers import init_st, load_model
from sgm.modules.diffusionmodules.sampling import EulerAncestralSampler
from models import VideoCompressionModel
from dataset import VideoFramePairDataset
from train import SubstepSampler, seeded_randn, SeededNoise, VERSION2SPECS


class VideoReconstructor:
    """Reconstruct video frames using trained compression model."""
    
    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda",
        diffusion_steps: int = 1
    ):
        self.device = device
        
        # Load checkpoint
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        args = checkpoint['args']
        
        # Initialize compression model
        self.model = VideoCompressionModel(
            clip_arch=args.clip_arch,
            clip_version=args.clip_version,
            hidden_dim=args.hidden_dim,
            seq_len=77,
            embed_dim=1024,
            rank=args.rank,
            freeze_encoder=True,
            device=device
        ).to(device)
        
        # Load trained weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Initialize SD-Turbo
        print("Loading SD-Turbo model...")
        self._init_sd_model(diffusion_steps)
        
        # Noise settings
        self.sigma = torch.Tensor([args.noise_sigma]).float().to(device)
        self.seed = args.seed
        
    def _init_sd_model(self, diffusion_steps: int):
        """Initialize SD-Turbo model and sampler."""
        version_dict = VERSION2SPECS['SD-Turbo']
        
        if not os.path.exists(version_dict["ckpt"]):
            print(f"Warning: SD-Turbo checkpoint not found at {version_dict['ckpt']}")
            self.sd_model = None
            self.sampler = None
            self.decoder = None
            return
            
        state = init_st(version_dict, load_filter=False)
        self.sd_model = state["model"]
        load_model(self.sd_model)
        
        # Use TinyVAE decoder for faster inference
        try:
            self.taesd = AutoencoderTiny.from_pretrained(
                "madebyollin/taesd", 
                torch_dtype=torch.float32
            ).cuda()
            self.decoder = self.taesd.decoder
        except Exception:
            self.decoder = None
            self.taesd = None
        
        # Initialize sampler
        self.sampler = SubstepSampler(
            n_sample_steps=diffusion_steps,
            num_steps=1000,
            eta=1.0,
            discretization_config=dict(
                target="sgm.modules.diffusionmodules.discretizer.LegacyDDPMDiscretization"
            ),
        )
        self.sampler.noise_sampler = SeededNoise(seed=self.seed)
    
    def denoiser(self, input, sigma, c):
        """Denoiser function for diffusion."""
        return self.sd_model.denoiser(
            self.sd_model.model,
            input,
            sigma,
            c,
        )
    
    def encode_to_latent(self, image: torch.Tensor) -> torch.Tensor:
        """Encode image to VAE latent space."""
        if self.sd_model is None:
            batch_size = image.shape[0]
            return torch.randn(batch_size, 4, 64, 64, device=image.device)
        return self.sd_model.encode_first_stage(image)
    
    def decode_from_latent(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode from VAE latent space to image."""
        if self.decoder is not None:
            return self.decoder(latent)
        elif self.sd_model is not None:
            return self.sd_model.decode_first_stage(latent)
        else:
            batch_size = latent.shape[0]
            return torch.randn(batch_size, 3, 512, 512, device=latent.device)
    
    @torch.no_grad()
    def reconstruct(
        self, 
        img_a: torch.Tensor, 
        img_b: torch.Tensor
    ) -> torch.Tensor:
        """
        Reconstruct image B given reference image A.
        
        Args:
            img_a: Reference image (batch, 3, H, W) in range [-1, 1]
            img_b: Target image (used for warm start) in range [-1, 1]
            
        Returns:
            Reconstructed image in range [-1, 1]
        """
        img_a = img_a.to(self.device)
        img_b = img_b.to(self.device)
        
        # Generate conditions
        outputs = self.model(img_a, img_b)
        condition = outputs['condition']
        null_text = outputs['null_text']
        
        # Encode target and add noise for warm start
        target_latent = self.encode_to_latent(img_b)
        noise = seeded_randn(target_latent.shape, self.seed)
        noised_latent = target_latent * self.sigma + noise * (1 - self.sigma)
        
        # Run diffusion
        if self.sd_model is not None:
            c = {'crossattn': condition}
            uc = {'crossattn': null_text}
            pred_latent = self.sampler(
                self.denoiser, 
                noised_latent, 
                cond=c, 
                uc=uc
            )
            
            # Decode to image
            pred_image = self.decode_from_latent(pred_latent)
        else:
            pred_image = img_b
        
        return pred_image
    
    @torch.no_grad()
    def get_compressed_representation(
        self,
        img_a: torch.Tensor,
        img_b: torch.Tensor
    ) -> dict:
        """
        Get compressed representation (U, V) for transmission.
        
        Args:
            img_a: Reference image
            img_b: Target image
            
        Returns:
            Dictionary with U, V tensors for transmission
        """
        img_a = img_a.to(self.device)
        img_b = img_b.to(self.device)
        
        outputs = self.model(img_a, img_b)
        
        return {
            'U': outputs['U'].cpu(),
            'V': outputs['V'].cpu(),
            'rank': self.model.get_rank()
        }


def process_video(
    reconstructor: VideoReconstructor,
    frame_dir: str,
    output_dir: str,
    interval: int = 1,
    max_frames: int = None
):
    """
    Process a video sequence and save reconstructed frames.
    
    Args:
        reconstructor: VideoReconstructor instance
        frame_dir: Directory containing source frames
        output_dir: Directory to save reconstructed frames
        interval: Frame interval
        max_frames: Maximum frames to process
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create dataset
    dataset = VideoFramePairDataset(
        frame_dir=frame_dir,
        interval=interval,
        max_frames=max_frames,
        image_size=512
    )
    
    print(f"Processing {len(dataset)} frame pairs...")
    
    for idx in tqdm(range(len(dataset))):
        batch = dataset[idx]
        
        img_a = batch['img_a'].unsqueeze(0)
        img_b = batch['img_b'].unsqueeze(0)
        
        # Reconstruct
        pred = reconstructor.reconstruct(img_a, img_b)
        
        # Save
        pred_np = tensor_to_image(pred)
        frame_idx = batch['idx_b']
        cv2.imwrite(
            os.path.join(output_dir, f"{frame_idx:05d}.png"),
            pred_np
        )
    
    print(f"Saved reconstructed frames to {output_dir}")


def tensor_to_image(t: torch.Tensor) -> np.ndarray:
    """Convert tensor to numpy image for saving."""
    t = torch.clamp((t + 1) / 2, 0, 1)
    t = (t * 255).byte()
    t = t.permute(0, 2, 3, 1).cpu().numpy()
    return t[0][:, :, ::-1]  # RGB to BGR


def main():
    parser = argparse.ArgumentParser(
        description="Inference for Video Compression Model"
    )
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--frame_dir', type=str, required=True,
                        help='Directory containing video frames')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save reconstructed frames')
    parser.add_argument('--interval', type=int, default=1,
                        help='Frame interval')
    parser.add_argument('--max_frames', type=int, default=None,
                        help='Maximum frames to process')
    parser.add_argument('--diffusion_steps', type=int, default=1,
                        help='Number of diffusion steps')
    
    args = parser.parse_args()
    
    # Initialize reconstructor
    reconstructor = VideoReconstructor(
        checkpoint_path=args.checkpoint,
        diffusion_steps=args.diffusion_steps
    )
    
    # Process video
    process_video(
        reconstructor=reconstructor,
        frame_dir=args.frame_dir,
        output_dir=args.output_dir,
        interval=args.interval,
        max_frames=args.max_frames
    )


if __name__ == "__main__":
    main()
