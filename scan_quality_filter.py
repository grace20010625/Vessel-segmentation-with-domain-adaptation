"""
Scan quality filter for multi-layer NIfTI volumes.
Computes inter-slice correlation for each volume and outputs a filtered list.

Usage:
    python scan_quality_filter.py --data_dir /path/to/nii_gz_files --threshold 0.80

Output:
    - scan_quality_report.csv   : per-volume stats
    - accepted_scans.txt        : filenames passing threshold
    - rejected_scans.txt        : filenames failing threshold
"""

import os
import gzip
import struct
import argparse
import numpy as np
import csv
from pathlib import Path


def read_nii_gz(path):
    """Read a .nii.gz file without nibabel. Returns (data_array, pixdim)."""
    with gzip.open(path, 'rb') as f:
        header_bytes = f.read(348)

    dim      = struct.unpack_from('<8h', header_bytes, 40)
    pixdim   = struct.unpack_from('<8f', header_bytes, 76)
    datatype = struct.unpack_from('<h',  header_bytes, 70)[0]
    vox_off  = struct.unpack_from('<f',  header_bytes, 108)[0]

    dtype_map = {
        2: np.uint8, 4: np.int16, 8: np.int32,
        16: np.float32, 64: np.float64,
        256: np.int8, 512: np.uint16, 768: np.uint32,
    }
    np_dtype = dtype_map.get(datatype, np.float32)

    ndim  = dim[0]
    shape = tuple(dim[1:ndim + 1])

    with gzip.open(path, 'rb') as f:
        f.read(int(vox_off))
        raw = f.read()

    data = np.frombuffer(raw, dtype=np_dtype).reshape(shape)
    return data, pixdim


def inter_slice_correlation(data):
    """
    Compute mean Pearson correlation between adjacent slices (along Z axis).
    Returns (mean_corr, min_corr, max_corr, per_slice_corr_list).
    """
    n_slices = data.shape[2]
    if n_slices < 2:
        return 0.0, 0.0, 0.0, []

    corrs = []
    for z in range(n_slices - 1):
        a = data[:, :, z].flatten().astype(np.float64)
        b = data[:, :, z + 1].flatten().astype(np.float64)
        # Use corrcoef; handle edge case of zero std
        if a.std() < 1e-8 or b.std() < 1e-8:
            corrs.append(0.0)
        else:
            corrs.append(float(np.corrcoef(a, b)[0, 1]))

    return float(np.mean(corrs)), float(np.min(corrs)), float(np.max(corrs)), corrs


def volume_stats(data):
    """Basic intensity stats for a volume."""
    flat = data.flatten().astype(np.float64)
    return {
        'min':    int(flat.min()),
        'max':    int(flat.max()),
        'mean':   round(float(flat.mean()), 2),
        'std':    round(float(flat.std()), 2),
        'p1':     float(np.percentile(flat, 1)),
        'p99':    float(np.percentile(flat, 99)),
        'dark_pct':   round(100 * float((flat < 50).mean()), 2),   # likely background
        'sat_pct':    round(100 * float((flat >= 4090).mean()), 4), # saturated pixels
    }


def analyze_all(data_dir, threshold, extensions=('.nii.gz',)):
    data_dir = Path(data_dir)

    # Collect files
    files = []
    for ext in extensions:
        if ext == '.nii.gz':
            files += list(data_dir.glob('*.nii.gz'))
        else:
            files += list(data_dir.glob(f'*{ext}'))
    files = sorted(set(files))

    if not files:
        print(f"No NIfTI files found in {data_dir}")
        return

    print(f"Found {len(files)} files. Analyzing...")

    rows = []
    accepted = []
    rejected = []

    for i, fpath in enumerate(files, 1):
        print(f"  [{i:3d}/{len(files)}] {fpath.name} ...", end=' ', flush=True)
        try:
            data, pixdim = read_nii_gz(fpath)
            mean_c, min_c, max_c, _ = inter_slice_correlation(data)
            stats = volume_stats(data)

            n_slices   = data.shape[2]
            vox_xy     = round(float(pixdim[1]), 4)
            vox_z      = round(float(pixdim[3]), 4)
            passes     = mean_c >= threshold

            row = {
                'filename':     fpath.name,
                'shape':        f"{data.shape[0]}x{data.shape[1]}x{n_slices}",
                'n_slices':     n_slices,
                'vox_xy_mm':    vox_xy,
                'vox_z_mm':     vox_z,
                'mean_corr':    round(mean_c, 4),
                'min_corr':     round(min_c, 4),
                'max_corr':     round(max_c, 4),
                'passes':       'YES' if passes else 'NO',
                **stats,
            }
            rows.append(row)

            if passes:
                accepted.append(fpath.name)
                print(f"r={mean_c:.3f} ✓ PASS")
            else:
                rejected.append(fpath.name)
                print(f"r={mean_c:.3f} ✗ FAIL")

        except Exception as e:
            print(f"ERROR: {e}")
            rows.append({'filename': fpath.name, 'passes': 'ERROR', 'mean_corr': -1})
            rejected.append(fpath.name)

    # Write CSV report
    out_dir = data_dir
    csv_path = out_dir / 'scan_quality_report.csv'
    fieldnames = [
        'filename', 'passes', 'shape', 'n_slices', 'vox_xy_mm', 'vox_z_mm',
        'mean_corr', 'min_corr', 'max_corr',
        'min', 'max', 'mean', 'std', 'p1', 'p99', 'dark_pct', 'sat_pct',
    ]
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)

    # Write accepted / rejected lists
    (out_dir / 'accepted_scans.txt').write_text('\n'.join(accepted))
    (out_dir / 'rejected_scans.txt').write_text('\n'.join(rejected))

    # Summary
    print(f"\n{'='*55}")
    print(f"  Threshold : r >= {threshold}")
    print(f"  Total     : {len(files)}")
    print(f"  Accepted  : {len(accepted)}  ({100*len(accepted)/len(files):.1f}%)")
    print(f"  Rejected  : {len(rejected)}  ({100*len(rejected)/len(files):.1f}%)")

    if rows:
        all_corrs = [r['mean_corr'] for r in rows if isinstance(r['mean_corr'], float) and r['mean_corr'] >= 0]
        if all_corrs:
            print(f"\n  Correlation distribution across all volumes:")
            print(f"    mean = {np.mean(all_corrs):.3f}")
            print(f"    std  = {np.std(all_corrs):.3f}")
            print(f"    min  = {np.min(all_corrs):.3f}")
            print(f"    max  = {np.max(all_corrs):.3f}")
            # Histogram bins
            bins = [0.5, 0.6, 0.7, 0.75, 0.80, 0.85, 0.90, 0.95, 1.01]
            hist, _ = np.histogram(all_corrs, bins=bins)
            print(f"\n  Histogram:")
            for lo, hi, cnt in zip(bins[:-1], bins[1:], hist):
                bar = '█' * cnt
                print(f"    [{lo:.2f}, {hi:.2f})  {bar}  {cnt}")

    print(f"\n  Reports saved to:")
    print(f"    {csv_path}")
    print(f"    {out_dir / 'accepted_scans.txt'}")
    print(f"    {out_dir / 'rejected_scans.txt'}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NIfTI scan quality filter by inter-slice correlation')
    parser.add_argument('--data_dir',  type=str, required=True, help='Directory containing .nii.gz files')
    parser.add_argument('--threshold', type=float, default=0.80,
                        help='Minimum mean inter-slice correlation to accept (default: 0.80). '
                             'Set to 0 to just generate the report without filtering.')
    args = parser.parse_args()

    analyze_all(args.data_dir, args.threshold)
