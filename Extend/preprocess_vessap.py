"""
preprocess_vessap.py - Preprocess VesSAP data for cross-domain fine-tuning.

Usage:
    python preprocess_vessap.py
"""

import os
import json
import numpy as np
import nibabel as nib
from typing import Dict, List, Tuple
from config import DataConfig


# ============================================================================
# 1. CHANNEL FUSION
# ============================================================================

def fuse_channels(ch0: np.ndarray, ch1: np.ndarray, strategy: str) -> np.ndarray:
    """
    Fuse dual-channel VesSAP data into single-channel input.

    VesSAP uses two dyes:
      - ch0 (WGA): strong signal for small capillaries, weak for large vessels
      - ch1 (EB):  strong signal for large vessels, weak for capillaries
    """
    if strategy == "ch0_only":
        return ch0.copy()
    elif strategy == "ch1_only":
        return ch1.copy()
    elif strategy == "max_fusion":
        return np.maximum(ch0, ch1)
    elif strategy == "mean_fusion":
        return (ch0 + ch1) / 2.0
    else:
        raise ValueError(f"Unknown fusion strategy: {strategy}")


# ============================================================================
# 2. INTENSITY NORMALIZATION
# ============================================================================

def percentile_clip_normalize(volume: np.ndarray,
                               p_low: float = 0.5,
                               p_high: float = 99.5) -> np.ndarray:
    """Clip to [p_low, p_high] percentiles then rescale to [0, 1]."""
    v_low  = np.percentile(volume, p_low)
    v_high = np.percentile(volume, p_high)
    if v_high - v_low < 1e-8:
        return np.zeros_like(volume, dtype=np.float32)
    clipped    = np.clip(volume, v_low, v_high)
    normalized = (clipped - v_low) / (v_high - v_low)
    return normalized.astype(np.float32)


def histogram_match(source: np.ndarray,
                    ref_mean: float,
                    ref_std: float) -> np.ndarray:
    """Simple moment-matching to align to reference (MiniVess) statistics."""
    src_mean = source.mean()
    src_std  = source.std()
    if src_std < 1e-8:
        return np.full_like(source, ref_mean, dtype=np.float32)
    matched = (source - src_mean) / src_std * ref_std + ref_mean
    return np.clip(matched, 0, 1).astype(np.float32)


def normalize_volume(volume: np.ndarray, cfg: DataConfig) -> np.ndarray:
    volume = percentile_clip_normalize(
        volume, cfg.clip_percentile_low, cfg.clip_percentile_high)
    if cfg.use_histogram_matching:
        volume = histogram_match(volume, cfg.minivess_mean, cfg.minivess_std)
    return volume


# ============================================================================
# 3. PATCH EXTRACTION
# ============================================================================

def extract_patches(volume: np.ndarray,
                    label: np.ndarray,
                    patch_size: Tuple[int, int, int],
                    overlap: float = 0.25,
                    min_vessel_ratio: float = 0.001) -> List[Dict]:
    """Extract 3D patches with vessel-ratio filtering."""
    pD, pH, pW = patch_size
    D, H, W    = volume.shape

    stride_d = max(1, int(pD * (1 - overlap)))
    stride_h = max(1, int(pH * (1 - overlap)))
    stride_w = max(1, int(pW * (1 - overlap)))

    z_starts = list(range(0, max(D - pD + 1, 1), stride_d))
    y_starts = list(range(0, max(H - pH + 1, 1), stride_h))
    x_starts = list(range(0, max(W - pW + 1, 1), stride_w))

    # Ensure edges are covered
    if not z_starts or z_starts[-1] + pD < D:
        z_starts.append(max(0, D - pD))
    if not y_starts or y_starts[-1] + pH < H:
        y_starts.append(max(0, H - pH))
    if not x_starts or x_starts[-1] + pW < W:
        x_starts.append(max(0, W - pW))

    patches = []
    for z in z_starts:
        for y in y_starts:
            for x in x_starts:
                img_patch = volume[z:z+pD, y:y+pH, x:x+pW]
                lbl_patch = label [z:z+pD, y:y+pH, x:x+pW]

                # Pad if smaller than patch_size (edge case)
                if img_patch.shape != tuple(patch_size):
                    padded_img = np.zeros(patch_size, dtype=np.float32)
                    padded_lbl = np.zeros(patch_size, dtype=np.float32)
                    d, h, w = img_patch.shape
                    padded_img[:d, :h, :w] = img_patch
                    padded_lbl[:d, :h, :w] = lbl_patch
                    img_patch = padded_img
                    lbl_patch = padded_lbl

                vessel_ratio = lbl_patch.sum() / lbl_patch.size
                if vessel_ratio >= min_vessel_ratio:
                    patches.append({
                        'image':        img_patch.astype(np.float32),
                        'label':        lbl_patch.astype(np.float32),
                        'origin':       (z, y, x),
                        'vessel_ratio': float(vessel_ratio),
                    })

    return patches


# ============================================================================
# 4. MAIN PREPROCESSING PIPELINE
# ============================================================================

def preprocess_single_sample(
    ch0_path: str,
    ch1_path: str,
    label_path: str,
    cfg: DataConfig,
    channel_strategy: str,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Full preprocessing for one VesSAP sample."""

    ch0   = nib.load(ch0_path).get_fdata().astype(np.float32)
    ch1   = nib.load(ch1_path).get_fdata().astype(np.float32)
    label = nib.load(label_path).get_fdata().astype(np.float32)
    label = (label > 0).astype(np.float32)

    stats = {
        'ch0_range':    [float(ch0.min()),   float(ch0.max())],
        'ch1_range':    [float(ch1.min()),   float(ch1.max())],
        'ch0_mean':     float(ch0.mean()),
        'ch1_mean':     float(ch1.mean()),
        'label_fg_ratio': float(label.sum() / label.size),
        'channel_strategy': channel_strategy,
        'shape':        list(ch0.shape),
    }

    fused      = fuse_channels(ch0, ch1, channel_strategy)
    normalized = normalize_volume(fused, cfg)

    stats['normalized_mean'] = float(normalized.mean())
    stats['normalized_std']  = float(normalized.std())

    return normalized, label, stats


def run_preprocessing(cfg: DataConfig):
    """Run full preprocessing pipeline for all samples and strategies."""
    print("=" * 60)
    print("VesSAP Preprocessing Pipeline")
    print("=" * 60)

    all_metadata = {}

    for strategy in cfg.channel_strategies:
        print(f"\n--- Channel strategy: {strategy} ---")
        strategy_dir = os.path.join(cfg.processed_dir, strategy)
        os.makedirs(strategy_dir, exist_ok=True)

        for sample_idx, (ch0_fn, ch1_fn, lbl_fn) in enumerate(cfg.vessap_samples):
            sample_name = f"sample_{sample_idx + 1}"
            print(f"\nProcessing {sample_name}...")

            # Images come from extend_raw, labels from extend_seg
            ch0_path = os.path.join(cfg.vessap_data_dir,  ch0_fn)
            ch1_path = os.path.join(cfg.vessap_data_dir,  ch1_fn)
            lbl_path = os.path.join(cfg.vessap_label_dir, lbl_fn)

            print(f"  ch0  : {ch0_path}")
            print(f"  ch1  : {ch1_path}")
            print(f"  label: {lbl_path}")

            volume, label, stats = preprocess_single_sample(
                ch0_path, ch1_path, lbl_path, cfg, strategy)

            print(f"  Volume shape : {volume.shape}")
            print(f"  Normalised   : mean={stats['normalized_mean']:.4f}, "
                  f"std={stats['normalized_std']:.4f}")
            print(f"  Vessel ratio : {stats['label_fg_ratio']:.4f}")

            # Save full volume (for sliding-window inference in Exp A)
            sample_dir = os.path.join(strategy_dir, sample_name)
            os.makedirs(sample_dir, exist_ok=True)
            np.save(os.path.join(sample_dir, "volume.npy"), volume)
            np.save(os.path.join(sample_dir, "label.npy"),  label)

            # Extract and save patches (for training in Exp B & C)
            patches = extract_patches(
                volume, label,
                patch_size       = cfg.patch_size,
                overlap          = cfg.patch_overlap,
                min_vessel_ratio = cfg.min_vessel_ratio,
            )
            print(f"  Patches      : {len(patches)} (size={cfg.patch_size})")

            patch_dir = os.path.join(sample_dir, "patches")
            os.makedirs(patch_dir, exist_ok=True)
            for i, p in enumerate(patches):
                np.savez_compressed(
                    os.path.join(patch_dir, f"patch_{i:04d}.npz"),
                    image=p['image'],
                    label=p['label'],
                )

            stats['num_patches'] = len(patches)
            all_metadata[f"{strategy}/{sample_name}"] = stats

    # Save metadata
    metadata_path = os.path.join(cfg.processed_dir, "preprocessing_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(all_metadata, f, indent=2)
    print(f"\nMetadata saved → {metadata_path}")

    # Create leave-one-out split files
    _create_split_files(cfg)
    print("Preprocessing complete!")


def _create_split_files(cfg: DataConfig):
    """Create leave-one-out JSON split files."""
    n = len(cfg.vessap_samples)
    for strategy in cfg.channel_strategies:
        strategy_dir = os.path.join(cfg.processed_dir, strategy)
        for fold_idx in range(n):
            test_sample   = f"sample_{fold_idx + 1}"
            train_samples = [f"sample_{i+1}" for i in range(n) if i != fold_idx]
            split = {
                'fold':     fold_idx + 1,
                'train':    train_samples,
                'test':     [test_sample],
                'strategy': strategy,
            }
            with open(os.path.join(strategy_dir, f"fold_{fold_idx+1}.json"), 'w') as f:
                json.dump(split, f, indent=2)
    print(f"Created {n} leave-one-out folds per strategy")


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    cfg = DataConfig()
    run_preprocessing(cfg)
