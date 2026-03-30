"""
preprocess.py — Preprocessing pipeline for vessel segmentation
===============================================================
File structure expected:
    raw/
        mv01.nii.gz
        mv02.nii.gz
        ...
        seg/
            mv01_y.nii.gz
            mv02_y.nii.gz
            ...

Steps per volume:
  1. Sanity check  — shape match, label values, NaN/Inf, vessel fraction
  2. Clip          — cap image at p99.9 to remove scanner saturation artefacts
  3. Normalise     — Z-score (global mean/std of entire image volume)
  4. Resample      — optional, unify voxel spacing across volumes
                     image → trilinear (order=1), label → nearest neighbour (order=0)
  5. 2.5D windows  — sliding window along Z axis
                     image: (2k+1) channels, label: centre slice only

Usage:
    # Dry run — prints stats, writes nothing
    python preprocess.py --data_dir ./raw --out_dir ./processed --dry_run

    # Standard run, 3-channel window (k=1)
    python preprocess.py --data_dir ./raw --out_dir ./processed

    # 5-channel window + resample to 1mm isotropic
    python preprocess.py --data_dir ./raw --out_dir ./processed --window_k 2 --resample 1.0

    # Only process quality-filtered scans
    python preprocess.py --data_dir ./raw --out_dir ./processed --scan_list ./accepted_scans.txt

Output:
    processed/
        mv01_processed.npz   →  images (Z, C, H, W), labels (Z, H, W), indices, meta
        mv02_processed.npz
        ...
        preprocess_report.csv
"""

import csv
import gzip
import json
import struct
import argparse
from pathlib import Path
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom


# ─────────────────────────────────────────────────────────────────────────────
# I/O
# ─────────────────────────────────────────────────────────────────────────────

def read_nii_gz(path):
    img = nib.load(str(path))
    data = img.get_fdata(dtype=np.float32)
    hdr = img.header
    vox = hdr.get_zooms()
    return data, {'vox_xy': float(vox[0]), 'vox_z': float(vox[2])}

def find_pairs(data_dir, scan_list=None):
    """
    Find (image_path, label_path) pairs.
    Images: data_dir/mv*.nii.gz
    Labels: data_dir/seg/mv*_y.nii.gz
    """
    data_dir = Path(data_dir)
    seg_dir  = data_dir.parent / 'seg'

    if not seg_dir.exists():
        raise FileNotFoundError(f'seg/ subfolder not found at {seg_dir}')

    if scan_list:
        with open(scan_list) as f:
            names = [l.strip() for l in f if l.strip()]
        img_files = [data_dir / n for n in names if (data_dir / n).exists()]
    else:
        img_files = sorted(data_dir.glob('mv*.nii.gz'))

    pairs   = []
    missing = []
    for img_path in img_files:
        stem     = img_path.name.replace('.nii.gz', '').replace('.nii', '')
        lbl_path = seg_dir / f'{stem}_y.nii.gz'
        if lbl_path.exists():
            pairs.append((img_path, lbl_path))
        else:
            missing.append(img_path.name)

    if missing:
        print(f'WARNING — no label found for: {missing}')

    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Sanity check
# ─────────────────────────────────────────────────────────────────────────────

def sanity_check(image, label, name):
    warnings = []

    if image.shape != label.shape:
        raise ValueError(
            f'{name}: image shape {image.shape} != label shape {label.shape}'
        )

    n_nan = int(np.isnan(image).sum())
    n_inf = int(np.isinf(image).sum())
    if n_nan > 0:
        warnings.append(f'{n_nan} NaN in image — replaced with 0')
    if n_inf > 0:
        warnings.append(f'{n_inf} Inf in image — replaced with 0')

    unexpected = [v for v in np.unique(label).tolist() if v not in (0, 1)]
    if unexpected:
        warnings.append(f'unexpected label values {unexpected} — will binarise')

    vessel_pct = 100.0 * float((label == 1).mean())
    if vessel_pct < 0.01:
        warnings.append(f'vessel fraction very low: {vessel_pct:.4f}% — check label')
    if vessel_pct > 50.0:
        warnings.append(f'vessel fraction very high: {vessel_pct:.1f}% — check label')

    return warnings, vessel_pct


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Clip
# ─────────────────────────────────────────────────────────────────────────────

def clip_image(image, percentile=99.9):
    cap = float(np.percentile(image, percentile))
    return np.clip(image, image.min(), cap).astype(np.float32), cap


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Normalise
# ─────────────────────────────────────────────────────────────────────────────

def zscore_normalise(image):
    mu  = float(image.mean())
    std = float(image.std())
    if std < 1e-8:
        return image.copy(), mu, std
    return ((image - mu) / std).astype(np.float32), mu, std


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Resample
# ─────────────────────────────────────────────────────────────────────────────

def resample_pair(image, label, vox_xy, vox_z, target_spacing):
    zoom_xy = vox_xy / target_spacing
    zoom_z  = vox_z  / target_spacing

    if abs(zoom_xy - 1.0) < 0.02 and abs(zoom_z - 1.0) < 0.02:
        return image, label

    factors = (zoom_xy, zoom_xy, zoom_z)
    image_r = zoom(image, factors, order=1, prefilter=False).astype(np.float32)
    label_r = zoom(label, factors, order=0, prefilter=False)
    label_r = (label_r > 0).astype(np.uint8)
    return image_r, label_r


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — 2.5D sliding windows
# ─────────────────────────────────────────────────────────────────────────────

def make_windows(image, label, k=1, pad_mode='reflect'):
    """
    image  : (H, W, Z) float32
    label  : (H, W, Z) uint8

    Returns
    -------
    img_windows : (Z, 2k+1, H, W)  float32
    lbl_windows : (Z, H, W)         uint8   — centre slice label only
    indices     : list[int]
    """
    H, W, Z = image.shape

    img_pad = np.pad(image, [(0, 0), (0, 0), (k, k)], mode=pad_mode)
    lbl_pad = np.pad(label, [(0, 0), (0, 0), (k, k)], mode='edge')

    img_windows = np.zeros((Z, 2 * k + 1, H, W), dtype=np.float32)
    lbl_windows = np.zeros((Z, H, W),             dtype=np.uint8)

    for z in range(Z):
        for c, offset in enumerate(range(-k, k + 1)):
            img_windows[z, c] = img_pad[:, :, z + k + offset]
        lbl_windows[z] = lbl_pad[:, :, z + k]

    return img_windows, lbl_windows, list(range(Z))


# ─────────────────────────────────────────────────────────────────────────────
# Single volume
# ─────────────────────────────────────────────────────────────────────────────

def process_one(img_path, lbl_path, out_dir,
                target_spacing=None, window_k=1, clip_pct=99.9,
                dry_run=False):
    name = Path(img_path).name
    stem = name.replace('.nii.gz', '').replace('.nii', '')
    log  = {'filename': name}

    image, hdr = read_nii_gz(img_path)
    label, _   = read_nii_gz(lbl_path)
    label = (label > 0).astype(np.uint8)

    log['original_shape'] = 'x'.join(map(str, image.shape))
    log['vox_xy_mm']      = round(hdr['vox_xy'], 4)
    log['vox_z_mm']       = round(hdr['vox_z'],  4)

    # 1 — sanity check
    warnings, vessel_pct = sanity_check(image, label, name)
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    log['vessel_pct'] = round(vessel_pct, 4)
    log['warnings']   = ' | '.join(warnings) if warnings else 'none'

    # 2 — clip
    image, cap = clip_image(image, percentile=clip_pct)
    log['clip_cap'] = round(cap, 2)

    # 3 — normalise
    image, mu, std = zscore_normalise(image)
    log['norm_mean'] = round(mu, 4)
    log['norm_std']  = round(std, 4)

    # 4 — resample (optional)
    if target_spacing is not None:
        image, label = resample_pair(
            image, label, hdr['vox_xy'], hdr['vox_z'], target_spacing
        )
        log['shape_after_resample'] = 'x'.join(map(str, image.shape))

    # 5 — 2.5D windows
    img_win, lbl_win, indices = make_windows(image, label, k=window_k)
    log['n_samples']  = len(indices)
    log['n_channels'] = 2 * window_k + 1

    if not dry_run:
        out_path = Path(out_dir) / f'{stem}_processed.npz'
        np.savez_compressed(
            out_path,
            images  = img_win,
            labels  = lbl_win,
            indices = np.array(indices),
            meta    = np.array(json.dumps({
                'filename':  name,
                'vox_xy':    hdr['vox_xy'],
                'vox_z':     hdr['vox_z'],
                'norm_mean': mu,
                'norm_std':  std,
                'clip_cap':  cap,
                'window_k':  window_k,
            })),
        )

    return log


# ─────────────────────────────────────────────────────────────────────────────
# Batch
# ─────────────────────────────────────────────────────────────────────────────

def run_batch(data_dir, out_dir, scan_list=None,
              target_spacing=None, window_k=1, clip_pct=99.9, dry_run=False):

    pairs = find_pairs(data_dir, scan_list)
    if not pairs:
        print('No image/label pairs found.')
        return

    print(f'Found {len(pairs)} image/label pairs')
    print(f'  window_k={window_k} ({2*window_k+1} channels)  '
          f'resample={target_spacing}  clip_pct={clip_pct}  dry_run={dry_run}\n')

    if not dry_run:
        Path(out_dir).mkdir(parents=True, exist_ok=True)

    all_logs      = []
    total_samples = 0

    for i, (img_path, lbl_path) in enumerate(pairs, 1):
        print(f'[{i:3d}/{len(pairs)}] {img_path.name}', end=' ... ', flush=True)
        try:
            log = process_one(
                img_path, lbl_path, out_dir,
                target_spacing=target_spacing,
                window_k=window_k,
                clip_pct=clip_pct,
                dry_run=dry_run,
            )
            total_samples += log.get('n_samples', 0)
            warn_str = f'  WARN: {log["warnings"]}' if log['warnings'] != 'none' else ''
            print(f'ok — {log["n_samples"]} samples  vessel={log["vessel_pct"]:.3f}%{warn_str}')
        except Exception as e:
            print(f'ERROR: {e}')
            log = {'filename': img_path.name, 'warnings': f'FAILED: {e}'}

        all_logs.append(log)

    if not dry_run and all_logs:
        report_path = Path(out_dir) / 'preprocess_report.csv'
        with open(report_path, 'w', newline='') as f:
            writer = csv.DictWriter(
                f, fieldnames=list(all_logs[0].keys()), extrasaction='ignore'
            )
            writer.writeheader()
            writer.writerows(all_logs)
        print(f'\nReport → {report_path}')

    print(f'\nDone.  {len(pairs)} volumes,  {total_samples} total 2.5D samples')
    if not dry_run:
        print(f'Output → {Path(out_dir).resolve()}')


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Preprocessing pipeline for vessel segmentation'
    )
    parser.add_argument('--data_dir',  required=True,
                        help='Root folder with mv*.nii.gz and seg/ subfolder')
    parser.add_argument('--out_dir',   required=True,
                        help='Output folder for processed .npz files')
    parser.add_argument('--scan_list', default=None,
                        help='accepted_scans.txt to process a subset only')
    parser.add_argument('--window_k',  type=int, default=1,
                        help='Half-width of 2.5D window. k=1→3ch, k=2→5ch (default: 1)')
    parser.add_argument('--resample',  type=float, default=None,
                        help='Target isotropic voxel spacing in mm (e.g. 1.0). '
                             'Omit to skip resampling.')
    parser.add_argument('--clip_pct',  type=float, default=99.9,
                        help='Percentile for intensity clipping (default: 99.9)')
    parser.add_argument('--dry_run',   action='store_true',
                        help='Print stats only, do not write any files')
    args = parser.parse_args()

    run_batch(
        data_dir       = args.data_dir,
        out_dir        = args.out_dir,
        scan_list      = args.scan_list,
        target_spacing = args.resample,
        window_k       = args.window_k,
        clip_pct       = args.clip_pct,
        dry_run        = args.dry_run,
    )
