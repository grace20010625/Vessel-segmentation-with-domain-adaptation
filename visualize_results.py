"""
visualize_results.py — Visualise segmentation predictions for paper figures
============================================================================
Generates publication-quality comparison figures:
    Column 1: Input image (centre channel, ITK-SNAP window)
    Column 2: Ground truth label overlay
    Column 3: Predicted mask overlay
    Column 4: Error map (FP=red, FN=blue, TP=green)

Can be called:
    1. Standalone after training to visualise saved predictions
    2. During training (call save_prediction_figure()) to monitor progress

Usage:
    python visualize_results.py \
        --npz_dir  ./data/processed \
        --ckpt     ./checkpoints/best_model.pth \
        --out_dir  ./figures \
        --n_samples 8
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch


# ─────────────────────────────────────────────────────────────────────────────
# Display helpers
# ─────────────────────────────────────────────────────────────────────────────

# ITK-SNAP window parameters (discovered during data analysis)
ITK_LO = 142.0
ITK_HI = 4095.0


def to_display(img_slice, mu=None, std=None):
    """
    Convert a normalised image slice back to display range [0,1].
    If mu/std provided: reverse z-score then apply ITK window.
    Otherwise: simple min-max stretch.
    """
    sl = img_slice.copy().astype(np.float32)
    if mu is not None and std is not None:
        sl = sl * std + mu
        return np.clip((sl - ITK_LO) / (ITK_HI - ITK_LO), 0, 1)
    else:
        lo, hi = sl.min(), sl.max()
        if hi - lo < 1e-8:
            return np.zeros_like(sl)
        return np.clip((sl - lo) / (hi - lo), 0, 1)


def overlay(img_disp, mask, color, alpha=0.5):
    """Blend a binary mask onto a grayscale display image as a coloured overlay."""
    rgb = np.stack([img_disp] * 3, axis=-1)
    r, g, b = color
    m = mask.astype(bool)
    rgb[m, 0] = np.clip(rgb[m, 0] * (1 - alpha) + r * alpha, 0, 1)
    rgb[m, 1] = np.clip(rgb[m, 1] * (1 - alpha) + g * alpha, 0, 1)
    rgb[m, 2] = np.clip(rgb[m, 2] * (1 - alpha) + b * alpha, 0, 1)
    return rgb


def error_map(pred_bin, target_bin, img_disp):
    """
    Colour-coded error map on top of the image:
        TP = green
        FP = red
        FN = blue
        TN = greyscale (no overlay)
    """
    pred   = pred_bin.astype(bool)
    target = target_bin.astype(bool)

    tp = pred & target
    fp = pred & ~target
    fn = ~pred & target

    rgb = np.stack([img_disp] * 3, axis=-1)
    alpha = 0.6

    def blend(mask, r, g, b):
        rgb[mask, 0] = np.clip(rgb[mask, 0] * (1 - alpha) + r * alpha, 0, 1)
        rgb[mask, 1] = np.clip(rgb[mask, 1] * (1 - alpha) + g * alpha, 0, 1)
        rgb[mask, 2] = np.clip(rgb[mask, 2] * (1 - alpha) + b * alpha, 0, 1)

    blend(tp, 0.0, 0.9, 0.0)   # green
    blend(fp, 0.9, 0.0, 0.0)   # red
    blend(fn, 0.0, 0.3, 0.9)   # blue

    return rgb


# ─────────────────────────────────────────────────────────────────────────────
# Core figure function
# ─────────────────────────────────────────────────────────────────────────────

def save_prediction_figure(images, labels, preds, out_path,
                           mu=None, std=None, title='',
                           threshold=0.5, dpi=200):
    """
    Save a comparison figure for a batch of samples.

    Parameters
    ----------
    images   : np.ndarray (N, C, H, W)  — model input (normalised)
    labels   : np.ndarray (N, 1, H, W)  — ground truth binary
    preds    : np.ndarray (N, 1, H, W)  — model output logits or probabilities
    out_path : str | Path
    mu, std  : float  — z-score parameters for display (from npz meta)
    title    : str
    threshold: float  — binarisation threshold
    dpi      : int
    """
    N = len(images)
    k = images.shape[1] // 2   # centre channel index

    fig, axes = plt.subplots(N, 4, figsize=(12, 3.2 * N), facecolor='white')
    if N == 1:
        axes = axes[None, :]
    fig.subplots_adjust(hspace=0.04, wspace=0.04,
                        top=0.94, bottom=0.02, left=0.06, right=0.98)

    col_titles = ['Input', 'Ground truth', 'Prediction', 'Error map']
    for col, ct in enumerate(col_titles):
        axes[0, col].set_title(ct, fontsize=10, pad=6)

    # Convert predictions to probabilities if they look like logits
    if preds.min() < 0 or preds.max() > 1:
        preds = 1 / (1 + np.exp(-preds))   # sigmoid

    for row in range(N):
        img_disp  = to_display(images[row, k], mu, std)
        gt_bin    = (labels[row, 0] > 0.5).astype(np.float32)
        pred_bin  = (preds[row, 0] >= threshold).astype(np.float32)

        # Compute per-sample dice for annotation
        inter = (pred_bin * gt_bin).sum()
        denom = pred_bin.sum() + gt_bin.sum()
        dice  = (2 * inter + 1e-6) / (denom + 1e-6)

        panels = [
            img_disp,
            overlay(img_disp, gt_bin,   color=(0.2, 0.8, 0.2)),
            overlay(img_disp, pred_bin, color=(0.9, 0.4, 0.0)),
            error_map(pred_bin, gt_bin, img_disp),
        ]

        for col, panel in enumerate(panels):
            ax = axes[row, col]
            if panel.ndim == 2:
                ax.imshow(panel, cmap='gray', vmin=0, vmax=1,
                          aspect='equal', interpolation='nearest')
            else:
                ax.imshow(panel, aspect='equal', interpolation='nearest')
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor('#cccccc')
                sp.set_linewidth(0.5)

        # Dice score annotation on prediction panel
        axes[row, 2].text(
            0.98, 0.02, f'Dice={dice:.3f}',
            transform=axes[row, 2].transAxes,
            color='white', fontsize=7, ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.5),
        )

    # Legend for error map
    legend_patches = [
        mpatches.Patch(color=(0.0, 0.9, 0.0), label='TP'),
        mpatches.Patch(color=(0.9, 0.0, 0.0), label='FP'),
        mpatches.Patch(color=(0.0, 0.3, 0.9), label='FN'),
    ]
    fig.legend(handles=legend_patches, loc='lower right',
               fontsize=7, framealpha=0.8,
               bbox_to_anchor=(0.99, 0.005), ncol=3)

    if title:
        fig.suptitle(title, fontsize=11, y=0.97)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Learning curve figure
# ─────────────────────────────────────────────────────────────────────────────

def save_learning_curves(train_losses, val_losses, val_dices, out_path, dpi=150):
    """
    Save training/validation loss and Dice curves.

    Parameters
    ----------
    train_losses : list[float]  — per-epoch train loss
    val_losses   : list[float]  — per-epoch val loss
    val_dices    : list[float]  — per-epoch val Dice
    out_path     : str | Path
    """
    epochs = list(range(1, len(train_losses) + 1))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), facecolor='white')
    fig.subplots_adjust(wspace=0.3)

    # Loss curves
    ax1.plot(epochs, train_losses, label='Train loss', color='#2266cc', lw=1.5)
    ax1.plot(epochs, val_losses,   label='Val loss',   color='#cc4422',
             lw=1.5, linestyle='--')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
    ax1.set_title('Training & Validation Loss')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Dice curve
    ax2.plot(epochs, val_dices, color='#22aa55', lw=1.5)
    if val_dices:
        ax2.axhline(max(val_dices), color='#22aa55', lw=0.8,
                    linestyle=':', alpha=0.7, label=f'Best={max(val_dices):.4f}')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Dice')
    ax2.set_title('Validation Dice')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Standalone inference + visualisation
# ─────────────────────────────────────────────────────────────────────────────

def run_visualisation(npz_dir, ckpt_path, out_dir, n_samples=8,
                      device='cpu', threshold=0.5):
    """Load a checkpoint and visualise predictions on validation set."""
    from dataset import get_dataloaders
    from unet import UNet

    _, val_loader = get_dataloaders(
        npz_dir=npz_dir, fold=0, n_folds=5,
        batch_size=n_samples, num_workers=0,
    )

    # Load model
    checkpoint = torch.load(ckpt_path, map_location=device)
    in_ch = checkpoint.get('in_channels', 3)
    model = UNet(in_channels=in_ch, out_channels=1).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    images, labels = next(iter(val_loader))
    with torch.no_grad():
        logits = model(images.to(device)).cpu()

    # Try to get norm stats from first npz
    mu, std = None, None
    npz_paths = sorted(Path(npz_dir).glob('*_processed.npz'))
    if npz_paths:
        d = np.load(str(npz_paths[0]), allow_pickle=True)
        if 'meta' in d:
            try:
                meta = json.loads(str(d['meta']))
                mu, std = meta.get('norm_mean'), meta.get('norm_std')
            except Exception:
                pass

    out_path = Path(out_dir) / 'predictions.png'
    save_prediction_figure(
        images=images.numpy(),
        labels=labels.numpy(),
        preds=logits.numpy(),
        out_path=out_path,
        mu=mu, std=std,
        title=f'Predictions — {Path(ckpt_path).stem}',
        threshold=threshold,
    )
    print(f'Saved → {out_path}')


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz_dir',   required=True)
    parser.add_argument('--ckpt',      required=True)
    parser.add_argument('--out_dir',   default='./figures')
    parser.add_argument('--n_samples', type=int, default=8)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--device',    default='cpu')
    args = parser.parse_args()

    run_visualisation(
        npz_dir   = args.npz_dir,
        ckpt_path = args.ckpt,
        out_dir   = args.out_dir,
        n_samples = args.n_samples,
        device    = args.device,
        threshold = args.threshold,
    )
