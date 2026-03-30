"""
dataset.py — Dataset and DataLoader for 2.5D vessel segmentation
=================================================================
Loads preprocessed .npz files produced by preprocess.py.

Each .npz contains:
    images  : float32  (Z, C, H, W)  — multi-channel 2.5D windows
    labels  : uint8    (Z, H, W)     — binary vessel mask (centre slice)
    indices : int      (Z,)          — original slice indices
    meta    : JSON string            — norm stats, voxel spacing, etc.

Two dataset classes:
    VesselDataset2D   — loads single npz, returns one slice at a time
    VesselDataset25D  — same but explicitly exposes the multi-channel
                        2.5D structure; ready for extension to wider windows

Usage:
    from dataset import get_dataloaders
    train_loader, val_loader = get_dataloaders(
        npz_dir='./data/processed',
        fold=0,
        n_folds=5,
        batch_size=8,
        window_k=1,
    )
"""

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF


# ─────────────────────────────────────────────────────────────────────────────
# Augmentation helpers (geometry only — no intensity, already normalised)
# ─────────────────────────────────────────────────────────────────────────────

def random_crop(image, label, crop_size=256):
    """Random crop image and label to crop_size x crop_size."""
    _, H, W = image.shape
    if H <= crop_size and W <= crop_size:
        return image, label
    top  = random.randint(0, H - crop_size)
    left = random.randint(0, W - crop_size)
    image = image[:, top:top+crop_size, left:left+crop_size]
    label = label[top:top+crop_size, left:left+crop_size]
    return image, label


def augment(image, label, crop_size=None):
    """
    Apply identical random geometric augmentation to image and label.
    image : (C, H, W) tensor float32
    label : (H, W)   tensor float32
    crop_size: if set, randomly crop to this size before other augmentations
    """
    # Random crop (reduces memory usage significantly)
    if crop_size is not None:
        image, label = random_crop(image, label, crop_size)

    # Random horizontal flip
    if random.random() > 0.5:
        image = TF.hflip(image)
        label = TF.hflip(label.unsqueeze(0)).squeeze(0)

    # Random vertical flip
    if random.random() > 0.5:
        image = TF.vflip(image)
        label = TF.vflip(label.unsqueeze(0)).squeeze(0)

    # Random 90-degree rotation
    k = random.randint(0, 3)
    if k > 0:
        image = torch.rot90(image, k, dims=[1, 2])
        label = torch.rot90(label, k, dims=[0, 1])

    return image, label


# ─────────────────────────────────────────────────────────────────────────────
# Core dataset
# ─────────────────────────────────────────────────────────────────────────────

class VesselDataset25D(Dataset):
    """
    Loads one or more preprocessed .npz files and returns individual
    2.5D slice samples.

    Parameters
    ----------
    npz_paths : list[str | Path]
        Paths to *_processed.npz files to include in this split.
    augment   : bool
        Apply random geometric augmentation (train=True, val=False).
    window_k  : int
        Half-width of the 2.5D window expected in the npz.
        Used only for validation — actual channels come from the file.
    """

    def __init__(self, npz_paths, augment=False, window_k=1, crop_size=None):
        self.augment   = augment
        self.window_k  = window_k
        self.crop_size = crop_size
        self.samples   = []   # list of (image_slice, label_slice)

        for path in npz_paths:
            data    = np.load(str(path), allow_pickle=True)
            images  = data['images']   # (Z, C, H, W)  float32
            labels  = data['labels']   # (Z, H, W)     uint8

            for z in range(images.shape[0]):
                img = images[z]                          # (C, H, W)
                lbl = (labels[z] > 0).astype(np.float32)  # (H, W) binary float
                self.samples.append((img, lbl))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_np, lbl_np = self.samples[idx]

        image = torch.from_numpy(img_np.copy())           # (C, H, W)
        # 2D mode: only use centre channel
        if image.shape[0] > 1 and self.window_k == 0:
            k = image.shape[0] // 2
            image = image[k:k+1]
        label = torch.from_numpy(lbl_np.copy())           # (H, W)

        if self.augment:
            image, label = augment(image, label, crop_size=self.crop_size)

        return image, label.unsqueeze(0)   # label → (1, H, W)


# ─────────────────────────────────────────────────────────────────────────────
# Volume-level fold splitting (prevent data leakage)
# ─────────────────────────────────────────────────────────────────────────────

def get_npz_paths(npz_dir):
    """Return sorted list of all *_processed.npz paths."""
    return sorted(Path(npz_dir).glob('*_processed.npz'))


def split_folds(npz_dir, n_folds=5, seed=42):
    """
    Split volumes (not slices) into n_folds for cross-validation.
    Returns list of (train_paths, val_paths) tuples, one per fold.

    Splitting at volume level is critical — slices from the same volume
    must not appear in both train and val, as adjacent slices are nearly
    identical and would cause data leakage.
    """
    paths = get_npz_paths(npz_dir)
    rng   = random.Random(seed)
    paths_shuffled = paths.copy()
    rng.shuffle(paths_shuffled)

    folds = []
    for fold in range(n_folds):
        val_paths   = paths_shuffled[fold::n_folds]
        train_paths = [p for p in paths_shuffled if p not in val_paths]
        folds.append((train_paths, val_paths))

    return folds


def get_dataloaders(npz_dir, fold=0, n_folds=5, batch_size=8,
                    window_k=1, num_workers=2, seed=42, crop_size=None, use_all=False):
    """
    Build train and validation DataLoaders for a given fold.

    Parameters
    ----------
    npz_dir    : str | Path   — directory with *_processed.npz files
    fold       : int          — which fold to use as validation (0-indexed)
    n_folds    : int          — total number of folds
    batch_size : int
    window_k   : int          — 2.5D half-width (informational, not enforced)
    num_workers: int
    seed       : int

    Returns
    -------
    train_loader, val_loader
    """

    all_paths = get_npz_paths(npz_dir)
    
    if use_all:
        train_paths = all_paths
        val_paths   = all_paths   # val 也用全部，只是为了监控
    else:
        folds = split_folds(npz_dir, n_folds=n_folds, seed=seed)
        train_paths, val_paths = folds[fold]

    folds = split_folds(npz_dir, n_folds=n_folds, seed=seed)
    train_paths, val_paths = folds[fold]

    print(f'Fold {fold}/{n_folds}: '
          f'{len(train_paths)} train volumes, {len(val_paths)} val volumes')

    train_ds = VesselDataset25D(train_paths, augment=True,  window_k=window_k, crop_size=crop_size)
    val_ds   = VesselDataset25D(val_paths,   augment=False, window_k=window_k, crop_size=None)

    print(f'  Train samples: {len(train_ds)},  Val samples: {len(val_ds)}')

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader


# ─────────────────────────────────────────────────────────────────────────────
# Quick sanity check
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz_dir', required=True)
    parser.add_argument('--fold',    type=int, default=0)
    parser.add_argument('--n_folds', type=int, default=5)
    args = parser.parse_args()

    train_loader, val_loader = get_dataloaders(
        npz_dir=args.npz_dir, fold=args.fold, n_folds=args.n_folds,
        batch_size=4, num_workers=0,
    )

    images, labels = next(iter(train_loader))
    print(f'\nBatch check:')
    print(f'  images: {images.shape}  dtype={images.dtype}  '
          f'min={images.min():.3f}  max={images.max():.3f}')
    print(f'  labels: {labels.shape}  dtype={labels.dtype}  '
          f'unique={labels.unique().tolist()}')
    print('DataLoader OK.')
