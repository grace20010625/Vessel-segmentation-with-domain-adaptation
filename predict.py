"""
predict.py — Inference + evaluation on out-of-distribution volumes
==================================================================
Usage:
    python predict.py \
        --npz_dir  ./data/processed_rejected \
        --ckpt     ./runs_final/fold0/checkpoints/best.pth \
        --out_dir  ./results_ood
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from metrics import SegmentationMetrics


# ─────────────────────────────────────────────────────────────────────────────
# Display helpers
# ─────────────────────────────────────────────────────────────────────────────

def to_display(img_slice, mu=None, std=None):
    # reverse zscore then apply ITK-SNAP window
    sl = img_slice.copy().astype(np.float32)
    if mu is not None and std is not None:
        sl = sl * std + mu
    lo, hi = np.percentile(sl, 1), np.percentile(sl, 99)
    if hi - lo < 1e-8:
        return np.zeros_like(sl)
    return np.clip((sl - lo) / (hi - lo), 0, 1)


def overlay(img, mask, color, alpha=0.5):
    rgb = np.stack([img] * 3, axis=-1)
    r, g, b = color
    m = mask.astype(bool)
    rgb[m, 0] = np.clip(rgb[m, 0] * (1 - alpha) + r * alpha, 0, 1)
    rgb[m, 1] = np.clip(rgb[m, 1] * (1 - alpha) + g * alpha, 0, 1)
    rgb[m, 2] = np.clip(rgb[m, 2] * (1 - alpha) + b * alpha, 0, 1)
    return rgb


def error_map(pred, target, img):
    tp = pred & target
    fp = pred & ~target
    fn = ~pred & target
    rgb = np.stack([img] * 3, axis=-1)
    alpha = 0.6
    rgb[tp, 0] = np.clip(rgb[tp, 0] * (1-alpha) + 0.0 * alpha, 0, 1)
    rgb[tp, 1] = np.clip(rgb[tp, 1] * (1-alpha) + 0.9 * alpha, 0, 1)
    rgb[tp, 2] = np.clip(rgb[tp, 2] * (1-alpha) + 0.0 * alpha, 0, 1)
    rgb[fp, 0] = np.clip(rgb[fp, 0] * (1-alpha) + 0.9 * alpha, 0, 1)
    rgb[fp, 1] = np.clip(rgb[fp, 1] * (1-alpha) + 0.0 * alpha, 0, 1)
    rgb[fp, 2] = np.clip(rgb[fp, 2] * (1-alpha) + 0.0 * alpha, 0, 1)
    rgb[fn, 0] = np.clip(rgb[fn, 0] * (1-alpha) + 0.0 * alpha, 0, 1)
    rgb[fn, 1] = np.clip(rgb[fn, 1] * (1-alpha) + 0.3 * alpha, 0, 1)
    rgb[fn, 2] = np.clip(rgb[fn, 2] * (1-alpha) + 0.9 * alpha, 0, 1)
    return rgb


# ─────────────────────────────────────────────────────────────────────────────
# Per-volume figure
# ─────────────────────────────────────────────────────────────────────────────

def save_volume_figure(stem, images, labels, preds, out_path,
                       mu=None, std=None, n_slices=4, threshold=0.5):
    Z = images.shape[0]
    k = images.shape[1] // 2

    # Pick slices with most vessel pixels
    vessel_counts = [labels[z, 0].sum() for z in range(Z)]
    sorted_z = sorted(range(Z), key=lambda i: vessel_counts[i], reverse=True)
    idxs = sorted(sorted_z[:n_slices])

    fig, axes = plt.subplots(len(idxs), 4, figsize=(12, 3.2 * len(idxs)),
                             facecolor='white')
    if len(idxs) == 1:
        axes = axes[None, :]
    fig.subplots_adjust(hspace=0.04, wspace=0.04,
                        top=0.93, bottom=0.02, left=0.05, right=0.98)

    for col, title in enumerate(['Input', 'Ground truth', 'Prediction', 'Error map']):
        axes[0, col].set_title(title, fontsize=10, pad=6)

    for row, z in enumerate(idxs):
        img_disp  = to_display(images[z, k], mu, std)
        gt_bin    = (labels[z, 0] > 0.5)
        pred_prob = torch.sigmoid(torch.tensor(preds[z, 0])).numpy()
        pred_bin  = pred_prob >= threshold

        inter = (pred_bin * gt_bin).sum()
        denom = pred_bin.sum() + gt_bin.sum()
        dice  = (2 * inter + 1e-6) / (denom + 1e-6)

        panels = [
            img_disp,
            overlay(img_disp, gt_bin,   (0.2, 0.8, 0.2)),
            overlay(img_disp, pred_bin, (0.9, 0.4, 0.0)),
            error_map(pred_bin, gt_bin, img_disp),
        ]

        for col, panel in enumerate(panels):
            ax = axes[row, col]
            if panel.ndim == 2:
                ax.imshow(panel, cmap='gray', vmin=0, vmax=1,
                          aspect='equal', interpolation='nearest')
            else:
                ax.imshow(panel, aspect='equal', interpolation='nearest')
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor('#cccccc'); sp.set_linewidth(0.5)

        axes[row, 0].set_ylabel(f'z={z}', fontsize=8, color='#555555', labelpad=4)
        axes[row, 2].text(0.98, 0.02, f'Dice={dice:.3f}',
                          transform=axes[row, 2].transAxes,
                          color='white', fontsize=7, ha='right', va='bottom',
                          bbox=dict(boxstyle='round,pad=0.2',
                                    facecolor='black', alpha=0.5))

    patches = [
        mpatches.Patch(color=(0.0, 0.9, 0.0), label='TP'),
        mpatches.Patch(color=(0.9, 0.0, 0.0), label='FP'),
        mpatches.Patch(color=(0.0, 0.3, 0.9), label='FN'),
    ]
    fig.legend(handles=patches, loc='lower right', fontsize=7,
               framealpha=0.8, bbox_to_anchor=(0.99, 0.005), ncol=3)
    fig.suptitle(stem, fontsize=11, y=0.97)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    ckpt    = torch.load(args.ckpt, map_location=device)
    in_ch   = ckpt.get('in_channels', 3)
    base_ch = ckpt.get('base_channels', 64)

    try:
        from full_attention_unet import FullAttentionUNet
        model = FullAttentionUNet(in_channels=in_ch, out_channels=1,
                                   base_channels=base_ch).to(device)
        print('Using FullAttentionUNet')
    except Exception:
        from unet import UNet
        model = UNet(in_channels=in_ch, out_channels=1,
                     base_channels=base_ch).to(device)
        print('Using UNet')

    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f'Loaded: epoch={ckpt.get("epoch","?")}, '
          f'best_dice={ckpt.get("best_dice", 0):.4f}\n')

    # Find volumes
    npz_paths = sorted(Path(args.npz_dir).glob('*_processed.npz'))
    print(f'Found {len(npz_paths)} volumes in {args.npz_dir}\n')

    global_eval = SegmentationMetrics(compute_hd95=True, threshold=args.threshold)
    all_results = []

    for npz_path in npz_paths:
        stem = npz_path.name.replace('_processed.npz', '')
        print(f'  {stem} ...', end=' ', flush=True)

        d      = np.load(str(npz_path), allow_pickle=True)
        images = d['images']   # (Z, C, H, W)
        labels = d['labels']   # (Z, H, W)

        mu, std = None, None
        if 'meta' in d:
            try:
                meta = json.loads(str(d['meta']))
                mu   = meta.get('norm_mean')
                std  = meta.get('norm_std')
            except Exception:
                pass

        # Inference
        all_logits = []
        vol_eval   = SegmentationMetrics(compute_hd95=False, threshold=args.threshold)

        with torch.no_grad():
            for i in range(0, len(images), args.batch_size):
                batch_img = torch.from_numpy(images[i:i+args.batch_size]).to(device)
                batch_lbl = torch.from_numpy(
                    labels[i:i+args.batch_size, None].astype(np.float32)).to(device)
                logits = model(batch_img)
                all_logits.append(logits.cpu().numpy())
                vol_eval.update(logits, batch_lbl)
                global_eval.update(logits, batch_lbl)

        vol_metrics = vol_eval.compute()
        all_logits  = np.concatenate(all_logits, axis=0)

        print(f'Dice={vol_metrics["dice"]:.4f}  IoU={vol_metrics["iou"]:.4f}')
        all_results.append({'volume': stem, **vol_metrics})

        # Save figure
        save_volume_figure(
            stem      = stem,
            images    = images,
            labels    = labels[:, None].astype(np.float32),
            preds     = all_logits,
            out_path  = out_dir / 'figures' / f'{stem}.png',
            mu=mu, std=std,
            n_slices  = args.n_slices,
            threshold = args.threshold,
        )

    # ── Output ────────────────────────────────────────────────────────────────
    global_metrics = global_eval.compute()

    # 1. JSON
    with open(out_dir / 'ood_results.json', 'w') as f:
        json.dump({
            'checkpoint':     args.ckpt,
            'n_volumes':      len(npz_paths),
            'threshold':      args.threshold,
            'global_metrics': global_metrics,
            'per_volume':     all_results,
        }, f, indent=2)

    # 2. CSV (per volume, easy to open in Excel)
    fieldnames = ['volume', 'dice', 'iou', 'precision', 'recall', 'accuracy']
    with open(out_dir / 'ood_results.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(all_results)

    # 3. Summary txt (human-readable table)
    lines = [
        '=' * 60,
        f'OOD Generalization Test — {len(npz_paths)} volumes',
        f'Checkpoint: {args.ckpt}',
        '=' * 60,
        f'{"Volume":<20} {"Dice":>8} {"IoU":>8} {"Prec":>8} {"Rec":>8}',
        '-' * 60,
    ]
    for r in all_results:
        lines.append(
            f'{r["volume"]:<20} {r["dice"]:>8.4f} {r["iou"]:>8.4f} '
            f'{r["precision"]:>8.4f} {r["recall"]:>8.4f}')
    lines += [
        '=' * 60,
        f'{"MEAN":<20} {global_metrics["dice"]:>8.4f} '
        f'{global_metrics["iou"]:>8.4f} '
        f'{global_metrics["precision"]:>8.4f} '
        f'{global_metrics["recall"]:>8.4f}',
    ]
    if 'hd95' in global_metrics:
        lines.append(f'HD95 (mean): {global_metrics["hd95"]:.2f}')
    lines.append('=' * 60)

    summary = '\n'.join(lines)
    print('\n' + summary)
    with open(out_dir / 'ood_summary.txt', 'w') as f:
        f.write(summary)

    print(f'\nJSON    → {out_dir}/ood_results.json')
    print(f'CSV     → {out_dir}/ood_results.csv')
    print(f'Summary → {out_dir}/ood_summary.txt')
    print(f'Figures → {out_dir}/figures/')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz_dir',    required=True)
    parser.add_argument('--ckpt',       required=True)
    parser.add_argument('--out_dir',    default='./results_ood')
    parser.add_argument('--batch_size', type=int,   default=16)
    parser.add_argument('--threshold',  type=float, default=0.5)
    parser.add_argument('--n_slices',   type=int,   default=4)
    args = parser.parse_args()
    main(args)
