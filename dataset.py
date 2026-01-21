"""
Dataset for Video Frame Pairs.

Loads consecutive frame pairs from video sequences for training
the video prediction reconstruction compression model.
"""

import os
import glob
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Optional


class VideoFramePairDataset(Dataset):
    """
    Dataset for loading video frame pairs (A, B) where A is the reference
    frame and B is the target frame to be reconstructed.
    """
    
    def __init__(
        self,
        frame_dir: str,
        interval: int = 1,
        max_frames: Optional[int] = None,
        image_size: int = 512,
        random_offset: bool = False
    ):
        """
        Args:
            frame_dir: Directory containing video frames (00000.png, 00001.png, ...)
            interval: Interval between reference and target frames
            max_frames: Maximum number of frames to use (None = all)
            image_size: Size to resize images to
            random_offset: Whether to use random offsets during training
        """
        self.frame_dir = frame_dir
        self.interval = interval
        self.image_size = image_size
        self.random_offset = random_offset
        
        # Find all frame files
        self.frame_paths = sorted(glob.glob(os.path.join(frame_dir, "*.png")))
        
        if max_frames is not None:
            self.frame_paths = self.frame_paths[:max_frames]
            
        # Number of valid pairs (need frame A and frame B)
        self.num_pairs = len(self.frame_paths) - interval
        
        if self.num_pairs <= 0:
            raise ValueError(
                f"Not enough frames for interval {interval}. "
                f"Found {len(self.frame_paths)} frames."
            )
    
    def __len__(self) -> int:
        return self.num_pairs
    
    def load_and_preprocess(self, path: str) -> torch.Tensor:
        """
        Load and preprocess an image.
        
        Args:
            path: Path to image file
            
        Returns:
            Tensor of shape (3, H, W) in range [-1, 1]
        """
        # Read image
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Failed to load image: {path}")
            
        # BGR to RGB
        img = img[:, :, ::-1].copy()
        
        # Center crop to square if necessary
        H, W, C = img.shape
        if H != W:
            min_dim = min(H, W)
            top = (H - min_dim) // 2
            left = (W - min_dim) // 2
            img = img[top:top+min_dim, left:left+min_dim, :]
        
        # Resize to target size
        if img.shape[0] != self.image_size:
            img = cv2.resize(img, (self.image_size, self.image_size))
        
        # Normalize to [-1, 1]
        img = (img.astype(np.float32) / 255.0) * 2.0 - 1.0
        
        # Convert to tensor (C, H, W)
        img = torch.from_numpy(img).permute(2, 0, 1)
        
        return img
    
    def __getitem__(self, idx: int) -> dict:
        """
        Get a frame pair.
        
        Args:
            idx: Index of the pair
            
        Returns:
            Dictionary with:
                - 'img_a': Reference image (3, H, W)
                - 'img_b': Target image (3, H, W)
                - 'idx_a': Index of reference frame
                - 'idx_b': Index of target frame
        """
        # Frame A (reference)
        idx_a = idx
        
        # Frame B (target) - with possible random offset
        if self.random_offset and self.training:
            offset = np.random.randint(1, self.interval + 1)
        else:
            offset = self.interval
        idx_b = idx + offset
        
        # Load images
        img_a = self.load_and_preprocess(self.frame_paths[idx_a])
        img_b = self.load_and_preprocess(self.frame_paths[idx_b])
        
        return {
            'img_a': img_a,
            'img_b': img_b,
            'idx_a': idx_a,
            'idx_b': idx_b,
            'path_a': self.frame_paths[idx_a],
            'path_b': self.frame_paths[idx_b]
        }


class MultiVideoDataset(Dataset):
    """
    Dataset that combines multiple video sequences for training.
    """
    
    def __init__(
        self,
        video_dirs: List[str],
        interval: int = 1,
        max_frames_per_video: Optional[int] = None,
        image_size: int = 512
    ):
        """
        Args:
            video_dirs: List of directories containing video frames
            interval: Interval between reference and target frames
            max_frames_per_video: Maximum frames per video
            image_size: Size to resize images to
        """
        self.datasets = []
        self.cumulative_lengths = [0]
        
        for video_dir in video_dirs:
            if os.path.isdir(video_dir):
                try:
                    ds = VideoFramePairDataset(
                        frame_dir=video_dir,
                        interval=interval,
                        max_frames=max_frames_per_video,
                        image_size=image_size
                    )
                    self.datasets.append(ds)
                    self.cumulative_lengths.append(
                        self.cumulative_lengths[-1] + len(ds)
                    )
                except ValueError as e:
                    print(f"Skipping {video_dir}: {e}")
        
        if len(self.datasets) == 0:
            raise ValueError("No valid video directories found")
    
    def __len__(self) -> int:
        return self.cumulative_lengths[-1]
    
    def __getitem__(self, idx: int) -> dict:
        # Find which dataset this index belongs to
        for i, (start, end) in enumerate(
            zip(self.cumulative_lengths[:-1], self.cumulative_lengths[1:])
        ):
            if start <= idx < end:
                local_idx = idx - start
                return self.datasets[i][local_idx]
        
        raise IndexError(f"Index {idx} out of range")


def create_dataloader(
    frame_dir: str,
    interval: int = 1,
    batch_size: int = 4,
    num_workers: int = 4,
    shuffle: bool = True,
    max_frames: Optional[int] = None,
    image_size: int = 512
) -> DataLoader:
    """
    Create a DataLoader for video frame pairs.
    
    Args:
        frame_dir: Directory containing video frames
        interval: Interval between frames
        batch_size: Batch size
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle data
        max_frames: Maximum number of frames
        image_size: Size to resize images to
        
    Returns:
        DataLoader instance
    """
    dataset = VideoFramePairDataset(
        frame_dir=frame_dir,
        interval=interval,
        max_frames=max_frames,
        image_size=image_size
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
