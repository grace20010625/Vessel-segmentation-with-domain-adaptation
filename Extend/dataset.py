"""
dataset.py - PyTorch Dataset classes for VesSAP fine-tuning experiments.
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple


class VesSAPPatchDataset(Dataset):
    """
    Dataset for loading preprocessed VesSAP patches.
    Used for training (Exp B fine-tune, Exp C from scratch).
    """

    def __init__(
        self,
        processed_dir: str,
        strategy: str,
        sample_names: List[str],
        transform=None,
    ):
        """
        Args:
            processed_dir: Root directory of preprocessed data
            strategy: Channel fusion strategy (e.g., "ch0_only", "max_fusion")
            sample_names: List of sample names to include (e.g., ["sample_1"])
            transform: Optional MONAI/torchvision transforms
        """
        self.transform = transform
        self.patch_files = []

        for sample_name in sample_names:
            patch_dir = os.path.join(processed_dir, strategy, sample_name, "patches")
            if not os.path.exists(patch_dir):
                raise FileNotFoundError(f"Patch directory not found: {patch_dir}")
            for fn in sorted(os.listdir(patch_dir)):
                if fn.endswith(".npz"):
                    self.patch_files.append(os.path.join(patch_dir, fn))

        print(f"  Loaded {len(self.patch_files)} patches from {sample_names}")

    def __len__(self):
        return len(self.patch_files)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        data = np.load(self.patch_files[idx])
        image = data['image']  # (D, H, W)
        label = data['label']  # (D, H, W)

        # Add channel dim: (1, D, H, W) for single-channel input
        image = image[np.newaxis, ...]
        label = label[np.newaxis, ...]

        sample = {
            'image': torch.from_numpy(image).float(),
            'label': torch.from_numpy(label).float(),
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


class VesSAPVolumeDataset(Dataset):
    """
    Dataset for loading full VesSAP volumes.
    Used for inference/evaluation (sliding window).
    """

    def __init__(
        self,
        processed_dir: str,
        strategy: str,
        sample_names: List[str],
    ):
        self.volumes = []

        for sample_name in sample_names:
            sample_dir = os.path.join(processed_dir, strategy, sample_name)
            volume = np.load(os.path.join(sample_dir, "volume.npy"))
            label = np.load(os.path.join(sample_dir, "label.npy"))
            self.volumes.append({
                'image': torch.from_numpy(volume[np.newaxis, ...]).float(),  # (1, D, H, W)
                'label': torch.from_numpy(label[np.newaxis, ...]).float(),
                'name': sample_name,
            })

    def __len__(self):
        return len(self.volumes)

    def __getitem__(self, idx):
        return self.volumes[idx]


# ============================================================================
# DATA AUGMENTATION (MONAI-based)
# ============================================================================

def get_train_transforms(cfg):
    """
    Build MONAI-based transforms for training.
    
    Adjust these to match your MiniVess training pipeline.
    Using MONAI dictionary transforms for consistency.
    """
    try:
        from monai.transforms import (
            Compose,
            RandFlipd,
            RandRotate90d,
            RandScaleIntensityd,
            RandShiftIntensityd,
            ToTensord,
        )

        transforms = [
            RandFlipd(keys=["image", "label"], prob=cfg.aug_flip_prob, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=cfg.aug_flip_prob, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=cfg.aug_flip_prob, spatial_axis=2),
            RandRotate90d(keys=["image", "label"], prob=cfg.aug_rotate_prob, spatial_axes=(0, 1)),
            RandScaleIntensityd(keys=["image"], factors=cfg.aug_intensity_scale, prob=0.5),
            RandShiftIntensityd(keys=["image"], offsets=cfg.aug_intensity_shift, prob=0.5),
        ]

        return Compose(transforms)

    except ImportError:
        print("  [Warning] MONAI not installed. Using basic PyTorch augmentation.")
        return BasicAugmentation(cfg)


class BasicAugmentation:
    """Fallback augmentation without MONAI dependency."""

    def __init__(self, cfg):
        self.flip_prob = cfg.aug_flip_prob
        self.intensity_shift = cfg.aug_intensity_shift
        self.intensity_scale = cfg.aug_intensity_scale

    def __call__(self, sample):
        image = sample['image']
        label = sample['label']

        # Random flips
        for axis in [1, 2, 3]:  # Skip channel dim (0)
            if torch.rand(1).item() < self.flip_prob:
                image = torch.flip(image, [axis])
                label = torch.flip(label, [axis])

        # Random intensity augmentation (image only)
        if torch.rand(1).item() < 0.5:
            shift = (torch.rand(1).item() * 2 - 1) * self.intensity_shift
            image = image + shift

        if torch.rand(1).item() < 0.5:
            scale = 1.0 + (torch.rand(1).item() * 2 - 1) * self.intensity_scale
            image = image * scale

        image = torch.clamp(image, 0, 1)

        return {'image': image, 'label': label}


# ============================================================================
# DATALOADER HELPERS
# ============================================================================

def create_dataloaders(
    processed_dir: str,
    strategy: str,
    fold: int,
    cfg,  # TrainingConfig
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and test dataloaders for a specific fold and strategy.
    
    Returns:
        (train_loader, test_loader)
    """
    # Load fold split
    split_path = os.path.join(processed_dir, strategy, f"fold_{fold}.json")
    with open(split_path) as f:
        split = json.load(f)

    train_samples = split['train']
    test_samples = split['test']

    # Train: patch-based with augmentation
    train_transform = get_train_transforms(cfg) if cfg.use_augmentation else None
    train_dataset = VesSAPPatchDataset(
        processed_dir, strategy, train_samples, transform=train_transform
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # Test: full volume for sliding-window inference
    test_dataset = VesSAPVolumeDataset(processed_dir, strategy, test_samples)

    return train_loader, test_dataset
