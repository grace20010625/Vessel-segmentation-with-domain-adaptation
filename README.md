# DSA5106 Group 3 — Vessel Segmentation with U-Net

Reproduction and extension of U-Net (Ronneberger et al., 2015) for micro-CT vessel segmentation.

---

## Project Structure

```
autodl-tmp/
│
├── data/
│   ├── raw/                        # Original NIfTI volumes (mv01.nii.gz ... mv70.nii.gz)
│   │   ├── accepted_scans.txt      # 43 volumes passing quality filter (r >= 0.70)
│   │   └── rejected_scans.txt      # 27 low-quality volumes
│   ├── seg/                        # Ground truth labels (mv01_y.nii.gz ... mv70_y.nii.gz)
│   │
│   ├── processed/                  # Preprocessed .npz files (43 accepted volumes)
│   ├── processed_rejected/         # Preprocessed .npz files (27 rejected volumes, OOD test)
│
├── runs_2.5D/                      # 2.5D U-Net baseline (window_k=1, 3-channel)
├── runs_2d/                        # 2D U-Net baseline (window_k=0, 1-channel)
├── runs_interslice/                # U-Net + Inter-slice Attention at bottleneck
├── runs_attn/                      # U-Net + Attention Gate on skip connections
├── runs_full_attn/                 # U-Net + Attention Gate + Inter-slice Attention
├── runs_final/                     # Final model trained on all 43 volumes
│
├── results_ood/                    # OOD generalization test on 27 rejected volumes
│   ├── figures/                    # Per-volume prediction figures
│   ├── ood_results.csv             # Per-volume metrics (Excel-friendly)
│   ├── ood_summary.txt             # Human-readable summary table
│   └── ood_results.json            # Full results
│
├── scan_quality_filter.py          # Step 1: filter volumes by inter-slice correlation
├── preprocess.py                   # Step 2: clip + z-score normalisation → .npz
├── preprocess_to_tiff.py           # Export preprocessed data as TIFF for inspection
├── dataset.py                      # PyTorch Dataset and DataLoader (volume-level CV)
├── unet.py                         # Standard U-Net
├── interslice_unet.py              # U-Net + Inter-slice Attention
├── attention_unet.py               # U-Net + Attention Gate
├── full_attention_unet.py          # U-Net + Attention Gate + Inter-slice Attention
├── metrics.py                      # Dice, IoU, HD95, Precision, Recall + loss functions
├── train.py                        # Training script (single fold or 5-fold CV)
├── predict.py                      # Inference + evaluation on new volumes
├── visualize_results.py            # Prediction figures and learning curves
└── Extend                          # Try to do domain adaptation
```

---

## Experiments

All experiments use 5-fold cross-validation, split at **volume level** to prevent data leakage.

| Run folder | Model | Input | batch\_size | crop\_size | Notes |
|---|---|---|---|---|---|
| `runs_2d/` | U-Net | 1-ch (2D) | 32 | 256 | Baseline: single slice |
| `runs_2.5D/` | U-Net | 3-ch (2.5D, k=1) | 32 | 256 | Baseline: stacked adjacent slices |
| `runs_interslice/` | InterSliceUNet | 3-ch (2.5D, k=1) | 32 | 256 | Cross-attention at bottleneck |
| `runs_attn/` | AttentionUNet | 3-ch (2.5D, k=1) | 32 | 256 | Attention Gate on skip connections |
| `runs_full_attn/` | FullAttentionUNet | 3-ch (2.5D, k=1) | 16 | 256 | AG + Inter-slice Attention |
| `runs_final/` | FullAttentionUNet | 3-ch (2.5D, k=1) | 16 | 256 | Final model, trained on all 43 volumes |

Each run folder has the same structure:

```
runs_*/
├── fold0/
│   ├── checkpoints/
│   │   ├── best.pth        # Best model (highest val Dice)
│   │   └── latest.pth      # Last epoch (for resuming)
│   ├── learning_curves.png
│   ├── config.json
│   └── results.json        # Final metrics for this fold
├── fold1/ ... fold4/
└── cv_summary.json         # Mean ± std across all 5 folds
```

---

## Data Pipeline

### Step 1 — Quality filtering

```bash
python scan_quality_filter.py --data_dir ./data/raw --threshold 0.75
```

Computes inter-slice Pearson correlation for each volume. Volumes below r=0.70 are rejected (low 3D continuity → 2.5D context less useful).

- Accepted: 43 volumes → `accepted_scans.txt`
- Rejected: 27 volumes → `rejected_scans.txt` (used for OOD test)

### Step 2 — Preprocessing

```bash
python preprocess.py --data_dir ./data/raw --out_dir ./data/processed \
    --scan_list ./data/raw/accepted_scans.txt --window_k 1
```

Per volume:
1. **Clip** intensities at p99.9 (remove scanner saturation artefacts)
2. **Z-score normalise** using global mean/std
3. **2.5D sliding window** along Z axis → (Z, 2k+1, H, W) image windows + (Z, H, W) labels

Output: one `*_processed.npz` per volume.

---

## Training

```bash
# Single fold (quick validation)
python train.py --npz_dir ./data/processed --fold 0 \
    --epochs 100 --batch_size 16 --crop_size 256 --window_k 1

# 5-fold cross-validation (background)
nohup python train.py --npz_dir ./data/processed \
    --run_all_folds --epochs 100 --batch_size 16 --crop_size 256 --window_k 1 \
    --out_dir ./runs_full_attn > ./runs_full_attn/cv_all.log 2>&1 &

# Final model on all data
nohup python train.py --npz_dir ./data/processed \
    --use_all --epochs 50 --batch_size 16 --crop_size 256 --window_k 1 \
    --out_dir ./runs_final > ./runs_final/train.log 2>&1 &
```

---

## Evaluation

### Cross-validation results

```bash
cat ./runs_full_attn/cv_all.log | grep -E "Fold|Mean|Dice"
```

### OOD generalization test

```bash
python predict.py \
    --npz_dir ./data/processed_rejected \
    --ckpt    ./runs_final/fold0/checkpoints/best.pth \
    --out_dir ./results_ood
```

```bash
cat ./results_ood/ood_summary.txt
```

---

## Key Design Decisions

**Volume-level CV split** — slices from the same volume are never split across train/val, preventing data leakage from near-identical adjacent slices.

**2.5D input** — stacking slices [t-1, t, t+1] as channels lets the 2D U-Net passively see neighbouring context without full 3D convolutions.

**Inter-slice Attention** — at the bottleneck, cross-attention between current slice features (Q) and neighbour features (K, V) lets the model actively query what neighbouring slices know, beyond passive channel stacking.

**Attention Gate** — on each skip connection, spatial attention weights suppress background regions and focus the decoder on vessel locations, critical for sparse vessel data (~3–5% foreground pixels).

**Loss function** — Dice + BCE with pos\_weight=9.0 to handle class imbalance.
