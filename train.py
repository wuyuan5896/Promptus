"""
Training script for Video Prediction Reconstruction Compression.

This script trains the MappingNet to generate conditions for SD-Turbo
that can reconstruct frame B given reference frame A.

Training Pipeline:
1. Load image pair (A, B)
2. Encode A and B with CLIP Image Encoder
3. MappingNet generates:
   - U, V (low-rank condition factors)
   - null_text (unconditional embedding)
4. VAE encodes B to latent space
5. Add noise for warm start
6. SD-Turbo denoises with conditions
7. Loss: MSE in latent space + regularization
8. Every 100 iters: decode and compute LPIPS, PSNR
"""

import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image import PeakSignalNoiseRatio
import cv2

from diffusers import AutoencoderTiny
from omegaconf import OmegaConf

# Try to import streamlit helpers - may not be available in all environments
try:
    from scripts.demo.streamlit_helpers import init_st, load_model
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    init_st = None
    load_model = None
    print("Warning: streamlit_helpers not available. SD-Turbo loading may be limited.")
from sgm.modules.diffusionmodules.sampling import EulerAncestralSampler
from models import VideoCompressionModel
from dataset import VideoFramePairDataset, create_dataloader


# ============================================================================
# Configuration
# ============================================================================

VERSION2SPECS = {
    "SD-Turbo": {
        "H": 512,
        "W": 512,
        "C": 4,
        "f": 8,
        "is_legacy": False,
        "config": "configs/inference/sd_2_1.yaml",
        "ckpt": "checkpoints/sd_turbo.safetensors",
    },
}


class SubstepSampler(EulerAncestralSampler):
    """Custom sampler for few-step diffusion."""
    
    def __init__(self, n_sample_steps=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_sample_steps = n_sample_steps
        self.steps_subset = [0, 100, 200, 300, 1000]

    def prepare_sampling_loop(self, x, cond, uc=None, num_steps=None):
        sigmas = self.discretization(
            self.num_steps if num_steps is None else num_steps, 
            device=self.device
        )
        sigmas = sigmas[
            self.steps_subset[: self.n_sample_steps] + self.steps_subset[-1:]
        ]
        if uc is None:
            uc = cond
        x = x * torch.sqrt(1.0 + sigmas[0] ** 2.0)
        num_sigmas = len(sigmas)
        s_in = x.new_ones([x.shape[0]])
        return x, s_in, sigmas, num_sigmas, cond, uc


def seeded_randn(shape, seed, device="cuda"):
    """Generate reproducible random noise."""
    randn = np.random.RandomState(seed).randn(*shape)
    randn = torch.from_numpy(randn).to(device=device, dtype=torch.float32)
    return randn


class SeededNoise:
    """Noise sampler with reproducible seeds."""
    
    def __init__(self, seed):
        self.seed = seed

    def __call__(self, x):
        self.seed = self.seed + 1
        return seeded_randn(x.shape, self.seed)


# ============================================================================
# Loss Functions
# ============================================================================

class VideoCompressionLoss(nn.Module):
    """
    Combined loss for video compression training.
    
    Components:
    1. Latent MSE Loss: MSE between VAE-encoded B and diffusion output
    2. Condition Regularization: Regularize U, V, and condition matrices
    3. Perceptual Loss (optional): LPIPS for perceptual quality
    """
    
    def __init__(
        self,
        latent_weight: float = 1.0,
        reg_weight: float = 0.1,
        perceptual_weight: float = 0.0,
        device: str = "cuda"
    ):
        super().__init__()
        
        self.latent_weight = latent_weight
        self.reg_weight = reg_weight
        self.perceptual_weight = perceptual_weight
        
        self.mse_loss = nn.MSELoss()
        
        if perceptual_weight > 0:
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type='vgg'
            ).to(device)
        else:
            self.lpips = None
    
    def forward(
        self,
        pred_latent: torch.Tensor,
        target_latent: torch.Tensor,
        U: torch.Tensor,
        V: torch.Tensor,
        condition: torch.Tensor,
        pred_image: torch.Tensor = None,
        target_image: torch.Tensor = None
    ) -> dict:
        """
        Compute combined loss.
        
        Args:
            pred_latent: Predicted latent from diffusion
            target_latent: Target latent from VAE encoding of B
            U: Low-rank factor
            V: Low-rank factor  
            condition: Full condition matrix
            pred_image: Predicted image (optional, for perceptual loss)
            target_image: Target image (optional, for perceptual loss)
            
        Returns:
            Dictionary with loss components
        """
        losses = {}
        
        # 1. Latent MSE Loss
        latent_loss = self.mse_loss(pred_latent, target_latent)
        losses['latent_mse'] = latent_loss * self.latent_weight
        
        # 2. Condition Regularization
        # Encourage sparse, well-distributed condition values
        reg_loss = (
            torch.mean(torch.abs(condition)) +  # L1 regularization
            0.1 * torch.mean(condition ** 2)     # L2 regularization
        )
        losses['reg'] = reg_loss * self.reg_weight
        
        # 3. Perceptual Loss (if enabled and images provided)
        if self.lpips is not None and pred_image is not None and target_image is not None:
            # Clamp images to [-1, 1] range for LPIPS
            pred_clamped = torch.clamp(pred_image, -1, 1)
            target_clamped = torch.clamp(target_image, -1, 1)
            perceptual_loss = self.lpips(pred_clamped, target_clamped)
            losses['perceptual'] = perceptual_loss * self.perceptual_weight
        
        # Total loss
        total_loss = sum(losses.values())
        losses['total'] = total_loss
        
        return losses


# ============================================================================
# Evaluation Metrics
# ============================================================================

class MetricsComputer:
    """Compute evaluation metrics for reconstruction quality."""
    
    def __init__(self, device: str = "cuda"):
        self.lpips = LearnedPerceptualImagePatchSimilarity(
            net_type='vgg'
        ).to(device)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
        self.device = device
    
    @torch.no_grad()
    def compute(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor
    ) -> dict:
        """
        Compute LPIPS and PSNR metrics.
        
        Args:
            pred: Predicted image in range [-1, 1] or [0, 1]
            target: Target image in same range
            
        Returns:
            Dictionary with metrics
        """
        # Ensure tensors are on correct device
        pred = pred.to(self.device)
        target = target.to(self.device)
        
        # Clamp to valid range
        pred_clamped = torch.clamp(pred, -1, 1)
        target_clamped = torch.clamp(target, -1, 1)
        
        # LPIPS (expects [-1, 1] range)
        lpips_value = self.lpips(pred_clamped, target_clamped).item()
        
        # PSNR (expects [0, 1] range)
        pred_01 = (pred_clamped + 1) / 2
        target_01 = (target_clamped + 1) / 2
        psnr_value = self.psnr(pred_01, target_01).item()
        
        return {
            'lpips': lpips_value,
            'psnr': psnr_value
        }


# ============================================================================
# Training Loop
# ============================================================================

class Trainer:
    """Main trainer class for video compression model."""
    
    def __init__(self, args):
        self.args = args
        self.device = "cuda"
        
        # Initialize SD-Turbo
        print("Loading SD-Turbo model...")
        self.init_sd_model()
        
        # Initialize compression model
        print("Initializing compression model...")
        self.compression_model = VideoCompressionModel(
            clip_arch=args.clip_arch,
            clip_version=args.clip_version,
            hidden_dim=args.hidden_dim,
            seq_len=77,
            embed_dim=1024,
            rank=args.rank,
            freeze_encoder=True,
            device=self.device
        ).to(self.device)
        
        # Initialize loss function
        self.loss_fn = VideoCompressionLoss(
            latent_weight=args.latent_weight,
            reg_weight=args.reg_weight,
            perceptual_weight=args.perceptual_weight,
            device=self.device
        )
        
        # Initialize metrics
        self.metrics = MetricsComputer(device=self.device)
        
        # Optimizer (only train MappingNet)
        self.optimizer = torch.optim.AdamW(
            self.compression_model.get_trainable_parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=args.num_epochs * args.steps_per_epoch,
            eta_min=args.lr * 0.01
        )
        
        # Gradient scaler for mixed precision
        self.scaler = GradScaler()
        
        # Tensorboard writer
        self.writer = SummaryWriter(log_dir=args.log_dir)
        
        # Noise settings
        self.sigma = torch.Tensor([args.noise_sigma]).float().cuda()
        self.seed = args.seed
        
        # Create output directories
        os.makedirs(args.log_dir, exist_ok=True)
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        os.makedirs(args.sample_dir, exist_ok=True)
        
    def init_sd_model(self):
        """Initialize SD-Turbo model and sampler."""
        version_dict = VERSION2SPECS['SD-Turbo']
        
        # Check if checkpoint exists
        if not os.path.exists(version_dict["ckpt"]):
            print(f"Warning: SD-Turbo checkpoint not found at {version_dict['ckpt']}")
            print("Please download the SD-Turbo model:")
            print("  1. Visit https://huggingface.co/stabilityai/sd-turbo")
            print("  2. Download 'sd_turbo.safetensors'")
            print(f"  3. Place it in the '{os.path.dirname(version_dict['ckpt'])}' folder")
            self.sd_model = None
            self.sampler = None
            self.decoder = None
            return
        
        # Check if streamlit helpers are available
        if not STREAMLIT_AVAILABLE:
            print("Warning: streamlit_helpers not available. Cannot load SD-Turbo.")
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
        except Exception as e:
            print(f"TinyVAE not available, using SD's VAE: {e}")
            self.decoder = None
            self.taesd = None
        
        # Initialize sampler
        self.sampler = SubstepSampler(
            n_sample_steps=self.args.diffusion_steps,
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
            # Return dummy latent for testing without SD model
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
            # Return dummy image for testing without SD model
            batch_size = latent.shape[0]
            return torch.randn(batch_size, 3, 512, 512, device=latent.device)
    
    def add_noise(self, latent: torch.Tensor, seed: int) -> torch.Tensor:
        """Add noise to latent for warm start."""
        noise = seeded_randn(latent.shape, seed)
        noised_latent = latent * self.sigma + noise * (1 - self.sigma)
        return noised_latent
    
    def train_step(
        self, 
        img_a: torch.Tensor, 
        img_b: torch.Tensor,
        step: int
    ) -> dict:
        """
        Single training step.
        
        Args:
            img_a: Reference image (batch, 3, H, W)
            img_b: Target image (batch, 3, H, W)
            step: Current step number
            
        Returns:
            Dictionary with losses
        """
        img_a = img_a.to(self.device)
        img_b = img_b.to(self.device)
        
        self.optimizer.zero_grad()
        
        with autocast(device_type='cuda', dtype=torch.float16):
            # 1. Generate conditions from compression model
            outputs = self.compression_model(img_a, img_b)
            U = outputs['U']
            V = outputs['V']
            condition = outputs['condition']
            null_text = outputs['null_text']
            
            # 2. Encode target image to latent space
            with torch.no_grad():
                target_latent = self.encode_to_latent(img_b)
            
            # 3. Add noise for warm start
            noised_latent = self.add_noise(target_latent, self.seed + step)
            
            # 4. Run diffusion with conditions
            if self.sd_model is not None:
                c = {'crossattn': condition}
                uc = {'crossattn': null_text}
                
                with torch.no_grad():
                    pred_latent = self.sampler(
                        self.denoiser, 
                        noised_latent, 
                        cond=c, 
                        uc=uc
                    )
            else:
                # Dummy output for testing without SD model
                pred_latent = noised_latent
            
            # 5. Compute loss
            losses = self.loss_fn(
                pred_latent=pred_latent,
                target_latent=target_latent,
                U=U,
                V=V,
                condition=condition
            )
        
        # Backward pass with gradient scaling
        self.scaler.scale(losses['total']).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(
            self.compression_model.get_trainable_parameters(), 
            max_norm=1.0
        )
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        
        return {k: v.item() for k, v in losses.items()}
    
    @torch.no_grad()
    def evaluate(
        self, 
        img_a: torch.Tensor, 
        img_b: torch.Tensor,
        step: int
    ) -> dict:
        """
        Evaluate reconstruction quality with LPIPS and PSNR.
        
        Args:
            img_a: Reference image
            img_b: Target image
            step: Current step for logging
            
        Returns:
            Dictionary with metrics
        """
        img_a = img_a.to(self.device)
        img_b = img_b.to(self.device)
        
        self.compression_model.eval()
        
        # Generate conditions
        outputs = self.compression_model(img_a, img_b)
        condition = outputs['condition']
        null_text = outputs['null_text']
        
        # Encode and add noise
        target_latent = self.encode_to_latent(img_b)
        noised_latent = self.add_noise(target_latent, self.seed)
        
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
            pred_image = img_b  # Dummy for testing
        
        # Compute metrics
        metrics = self.metrics.compute(pred_image, img_b)
        
        # Save sample images
        self.save_samples(img_a, img_b, pred_image, step)
        
        self.compression_model.train()
        
        return metrics
    
    def save_samples(
        self, 
        img_a: torch.Tensor, 
        img_b: torch.Tensor, 
        pred: torch.Tensor,
        step: int
    ):
        """Save sample images for visualization."""
        # Convert to numpy images
        def tensor_to_image(t):
            t = torch.clamp((t + 1) / 2, 0, 1)
            t = (t * 255).byte()
            t = t.permute(0, 2, 3, 1).cpu().numpy()
            return t[0][:, :, ::-1]  # RGB to BGR
        
        img_a_np = tensor_to_image(img_a)
        img_b_np = tensor_to_image(img_b)
        pred_np = tensor_to_image(pred)
        
        # Concatenate horizontally
        combined = np.concatenate([img_a_np, img_b_np, pred_np], axis=1)
        
        # Save
        cv2.imwrite(
            os.path.join(self.args.sample_dir, f"step_{step:06d}.png"),
            combined
        )
    
    def save_checkpoint(self, epoch: int, step: int):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': self.compression_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'args': self.args
        }
        path = os.path.join(
            self.args.checkpoint_dir, 
            f"checkpoint_epoch{epoch}_step{step}.pth"
        )
        torch.save(checkpoint, path)
        print(f"Saved checkpoint: {path}")
    
    def train(self, dataloader):
        """Main training loop."""
        print("Starting training...")
        
        global_step = 0
        
        for epoch in range(self.args.num_epochs):
            epoch_losses = []
            
            for batch_idx, batch in enumerate(dataloader):
                img_a = batch['img_a']
                img_b = batch['img_b']
                
                # Training step
                losses = self.train_step(img_a, img_b, global_step)
                epoch_losses.append(losses['total'])
                
                # Logging
                if global_step % self.args.log_interval == 0:
                    print(
                        f"Epoch {epoch}, Step {global_step}: "
                        f"Loss = {losses['total']:.4f}, "
                        f"Latent MSE = {losses['latent_mse']:.4f}, "
                        f"Reg = {losses['reg']:.4f}"
                    )
                    
                    for k, v in losses.items():
                        self.writer.add_scalar(f"train/{k}", v, global_step)
                    
                    self.writer.add_scalar(
                        "train/lr", 
                        self.scheduler.get_last_lr()[0], 
                        global_step
                    )
                
                # Evaluation every 100 steps
                if global_step % self.args.eval_interval == 0 and global_step > 0:
                    print(f"\nEvaluating at step {global_step}...")
                    metrics = self.evaluate(img_a, img_b, global_step)
                    print(
                        f"LPIPS = {metrics['lpips']:.4f}, "
                        f"PSNR = {metrics['psnr']:.2f} dB"
                    )
                    
                    self.writer.add_scalar("eval/lpips", metrics['lpips'], global_step)
                    self.writer.add_scalar("eval/psnr", metrics['psnr'], global_step)
                
                global_step += 1
            
            # End of epoch
            avg_loss = np.mean(epoch_losses)
            print(f"\nEpoch {epoch} completed. Average loss: {avg_loss:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % self.args.save_interval == 0:
                self.save_checkpoint(epoch, global_step)
        
        # Final checkpoint
        self.save_checkpoint(self.args.num_epochs - 1, global_step)
        print("Training completed!")
        
        self.writer.close()


# ============================================================================
# Main
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Video Prediction Reconstruction Compression Model"
    )
    
    # Data
    parser.add_argument('--frame_dir', type=str, default='data/sky',
                        help='Directory containing video frames')
    parser.add_argument('--max_frames', type=int, default=None,
                        help='Maximum number of frames to use')
    parser.add_argument('--interval', type=int, default=1,
                        help='Interval between reference and target frames')
    parser.add_argument('--image_size', type=int, default=512,
                        help='Image size')
    
    # Model
    parser.add_argument('--clip_arch', type=str, default='ViT-L-14',
                        help='CLIP architecture')
    parser.add_argument('--clip_version', type=str, default='openai',
                        help='CLIP pretrained version')
    parser.add_argument('--hidden_dim', type=int, default=1024,
                        help='Hidden dimension for MappingNet')
    parser.add_argument('--rank', type=int, default=8,
                        help='Rank for low-rank decomposition')
    parser.add_argument('--diffusion_steps', type=int, default=1,
                        help='Number of diffusion steps (1-4)')
    
    # Training
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--steps_per_epoch', type=int, default=100,
                        help='Steps per epoch (for scheduler)')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--noise_sigma', type=float, default=0.05,
                        help='Noise sigma for warm start')
    parser.add_argument('--seed', type=int, default=88,
                        help='Random seed')
    
    # Loss weights
    parser.add_argument('--latent_weight', type=float, default=1.0,
                        help='Weight for latent MSE loss')
    parser.add_argument('--reg_weight', type=float, default=0.1,
                        help='Weight for regularization loss')
    parser.add_argument('--perceptual_weight', type=float, default=0.0,
                        help='Weight for perceptual loss')
    
    # Logging
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Tensorboard log directory')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints_train',
                        help='Checkpoint save directory')
    parser.add_argument('--sample_dir', type=str, default='samples',
                        help='Sample image save directory')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Logging interval')
    parser.add_argument('--eval_interval', type=int, default=100,
                        help='Evaluation interval')
    parser.add_argument('--save_interval', type=int, default=10,
                        help='Checkpoint save interval (epochs)')
    
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create dataloader
    print(f"Loading data from {args.frame_dir}...")
    dataloader = create_dataloader(
        frame_dir=args.frame_dir,
        interval=args.interval,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        max_frames=args.max_frames,
        image_size=args.image_size
    )
    print(f"Dataset size: {len(dataloader.dataset)} pairs")
    
    # Create trainer
    trainer = Trainer(args)
    
    # Train
    trainer.train(dataloader)


if __name__ == "__main__":
    main()
