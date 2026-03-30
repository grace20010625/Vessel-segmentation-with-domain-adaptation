"""
run_experiments.py - VesSAP cross-domain fine-tuning with FullAttentionUNet.

Experiment A: Direct inference with pretrained FullAttentionUNet (no fine-tuning)
Experiment B: Fine-tune pretrained model on VesSAP
Experiment C: Train from scratch on VesSAP (lower-bound reference)

Usage:
    python run_experiments.py \
        --processed_dir ./Extend_data/vessap_processed \
        --pretrained_model ../runs_final/fold0/checkpoints/best.pth \
        --output_dir ./results
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from copy import deepcopy
from typing import Dict

# Add parent directory to import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from full_attention_unet import FullAttentionUNet
from config import ExperimentConfig
from dataset import create_dataloaders, VesSAPVolumeDataset
from metrics import compute_all_metrics, find_optimal_threshold

# CombinedLoss inline (local metrics.py is VesSAP metrics, not loss functions)
import torch.nn as nn
class CombinedLoss(nn.Module):
    def __init__(self, dice_weight=0.5, bce_weight=0.5, pos_weight=None):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight  = bce_weight
        self.pos_weight  = pos_weight
    def forward(self, logits, targets):
        import torch.nn.functional as F
        probs     = torch.sigmoid(logits)
        inter     = (probs * targets).sum(dim=(2, 3))
        denom     = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice_loss = (1.0 - (2.0 * inter + 1e-6) / (denom + 1e-6)).mean()
        pw        = torch.tensor([self.pos_weight], device=logits.device) \
                    if self.pos_weight else None
        bce_loss  = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pw)
        return self.dice_weight * dice_loss + self.bce_weight * bce_loss


# ============================================================================
# MODEL
# ============================================================================

def load_model(model_path: str, device: torch.device) -> nn.Module:
    model = FullAttentionUNet(in_channels=3, out_channels=1, base_channels=64)
    if model_path and os.path.exists(model_path):
        ckpt = torch.load(model_path, map_location=device)
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
            print(f"  Loaded: epoch={ckpt.get('epoch','?')}, "
                  f"best_dice={ckpt.get('best_dice', 0):.4f}")
        else:
            model.load_state_dict(ckpt)
    else:
        print(f"  [Warning] No pretrained model at {model_path}")
    return model.to(device)


def create_fresh_model(device: torch.device) -> nn.Module:
    model = FullAttentionUNet(in_channels=3, out_channels=1, base_channels=64)
    print("  Created fresh FullAttentionUNet (random init)")
    return model.to(device)


# ============================================================================
# PATCH → 2.5D SLICES
# ============================================================================

def patch_to_slices(images: torch.Tensor,
                    labels: torch.Tensor):
    """
    Convert 3D patches (B, 1, D, H, W) → 2.5D slice batches (B*D, 3, H, W).
    Each axial slice gets [t-1, t, t+1] as three channels.
    Label output is centre-slice only: (B*D, 1, H, W).
    """
    B, _, D, H, W = images.shape
    imgs_np = images.squeeze(1).cpu().numpy()   # (B, D, H, W)
    lbls_np = labels.squeeze(1).cpu().numpy()   # (B, D, H, W)

    slice_imgs = []
    slice_lbls = []

    for b in range(B):
        vol     = imgs_np[b]                    # (D, H, W)
        lbl     = lbls_np[b]
        vol_pad = np.pad(vol, [(1,1),(0,0),(0,0)], mode='reflect')  # (D+2, H, W)

        for z in range(D):
            prev = vol_pad[z]
            curr = vol_pad[z + 1]
            nxt  = vol_pad[z + 2]
            slice_imgs.append(np.stack([prev, curr, nxt], axis=0))  # (3, H, W)
            slice_lbls.append(lbl[z][None])                          # (1, H, W)

    img_tensor = torch.from_numpy(np.stack(slice_imgs)).float()
    lbl_tensor = torch.from_numpy(np.stack(slice_lbls)).float()
    return img_tensor, lbl_tensor


# ============================================================================
# 2.5D INFERENCE ON FULL VOLUME
# ============================================================================

def inference_2d5(model: nn.Module,
                  volume: np.ndarray,
                  device: torch.device,
                  batch_size: int = 8) -> np.ndarray:
    """
    Slice-by-slice 2.5D inference on a 3D volume.
    volume: float32 (D, H, W) already normalised.
    Returns prob_map: float32 (D, H, W) in [0, 1].
    """
    model.eval()
    D, H, W  = volume.shape
    prob_map = np.zeros((D, H, W), dtype=np.float32)
    vol_pad  = np.pad(volume, [(1,1),(0,0),(0,0)], mode='reflect')  # (D+2, H, W)

    with torch.no_grad():
        for start in range(0, D, batch_size):
            end   = min(start + batch_size, D)
            batch = []
            for z in range(start, end):
                prev = vol_pad[z]
                curr = vol_pad[z + 1]
                nxt  = vol_pad[z + 2]
                batch.append(np.stack([prev, curr, nxt], axis=0))  # (3, H, W)

            batch_tensor = torch.from_numpy(
                np.stack(batch)).float().to(device)                  # (B, 3, H, W)
            logits = model(batch_tensor)                             # (B, 1, H, W)
            probs  = torch.sigmoid(logits).squeeze(1).cpu().numpy() # (B, H, W)
            prob_map[start:end] = probs

    return prob_map


# ============================================================================
# TRAINING LOOP
# ============================================================================

def train_model(model, train_loader, loss_fn, optimizer, scheduler,
                num_epochs, device, patience=15, exp_name="") -> Dict:
    best_loss  = float('inf')
    best_state = None
    no_improve = 0
    history    = {'train_loss': [], 'epoch_time': []}

    print(f"\n  Training ({exp_name}): {num_epochs} epochs")

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        n_batches  = 0
        t0         = time.time()

        for batch in train_loader:
            images_3d = batch['image']   # (B, 1, D, H, W)
            labels_3d = batch['label']   # (B, 1, D, H, W)

            # Convert to 2.5D slices
            images_2d, labels_2d = patch_to_slices(images_3d, labels_3d)
            images_2d = images_2d.to(device)
            labels_2d = labels_2d.to(device)

            optimizer.zero_grad()
            logits = model(images_2d)
            loss   = loss_fn(logits, labels_2d)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches  += 1

        if scheduler:
            scheduler.step()

        avg_loss = epoch_loss / max(n_batches, 1)
        elapsed  = time.time() - t0
        history['train_loss'].append(avg_loss)
        history['epoch_time'].append(elapsed)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f"    Epoch {epoch+1:3d}/{num_epochs} | "
                  f"loss={avg_loss:.4f} | lr={lr:.2e} | {elapsed:.1f}s")

        if avg_loss < best_loss - 0.001:
            best_loss  = avg_loss
            best_state = deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"    Early stopping at epoch {epoch+1}")
                break

    if best_state:
        model.load_state_dict(best_state)

    history['best_loss']     = best_loss
    history['stopped_epoch'] = epoch + 1
    return history


# ============================================================================
# SHARED INFERENCE + EVAL
# ============================================================================

def _run_inference_and_eval(model, cfg, strategy, fold, device, save_dir):
    split_path = os.path.join(cfg.data.processed_dir, strategy, f"fold_{fold}.json")
    with open(split_path) as f:
        split = json.load(f)

    test_dataset = VesSAPVolumeDataset(
        cfg.data.processed_dir, strategy, split['test'])
    test_vol = test_dataset[0]

    volume = test_vol['image'].squeeze(0).numpy()   # (D, H, W)
    target = test_vol['label'].squeeze().numpy()    # (D, H, W)

    print("  Running 2.5D inference...")
    pred_prob = inference_2d5(model, volume, device,
                              batch_size=cfg.eval.sliding_window_batch_size)

    if cfg.eval.search_threshold:
        best_thresh, _, _ = find_optimal_threshold(
            pred_prob, target, cfg.eval.threshold_range)
        print(f"  Optimal threshold: {best_thresh:.2f}")
    else:
        best_thresh = cfg.eval.threshold

    pred_binary = (pred_prob > best_thresh).astype(bool)
    metrics = compute_all_metrics(
        pred_binary, target,
        compute_cldice = cfg.eval.compute_cldice,
        compute_hd95   = cfg.eval.compute_hausdorff,
    )
    metrics['threshold']   = best_thresh
    metrics['test_sample'] = split['test'][0]

    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, "pred_prob.npy"),   pred_prob)
    np.save(os.path.join(save_dir, "pred_binary.npy"), pred_binary)

    return metrics


# ============================================================================
# EXPERIMENT RUNNERS
# ============================================================================

def run_experiment_a(cfg, strategy, fold, device):
    print(f"\n{'='*60}")
    print(f"EXP A: Direct Inference | strategy={strategy} | fold={fold}")
    print(f"{'='*60}")

    model    = load_model(cfg.data.pretrained_model_path, device)
    save_dir = os.path.join(cfg.results_dir, f"exp_a/{strategy}/fold_{fold}")
    metrics  = _run_inference_and_eval(model, cfg, strategy, fold, device, save_dir)

    print(f"  Dice={metrics['dice']:.4f}  Prec={metrics['precision']:.4f}  "
          f"Rec={metrics['recall']:.4f}  clDice={metrics.get('cldice','N/A')}")
    return metrics


def run_experiment_b(cfg, strategy, fold, device):
    print(f"\n{'='*60}")
    print(f"EXP B: Fine-tune Pretrained | strategy={strategy} | fold={fold}")
    print(f"{'='*60}")

    model = load_model(cfg.data.pretrained_model_path, device)

    if cfg.training.finetune_freeze_encoder:
        print("  Freezing encoder...")
        for name, param in model.named_parameters():
            if 'enc' in name or 'bottleneck' in name:
                param.requires_grad = False

    train_loader, _ = create_dataloaders(
        cfg.data.processed_dir, strategy, fold, cfg.training)

    loss_fn   = CombinedLoss(dice_weight=0.5, bce_weight=0.5, pos_weight=9.0)
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.training.finetune_lr, weight_decay=cfg.training.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.training.finetune_epochs)

    history  = train_model(model, train_loader, loss_fn, optimizer, scheduler,
                           cfg.training.finetune_epochs, device,
                           cfg.training.patience, "fine-tune")

    save_dir = os.path.join(cfg.results_dir, f"exp_b/{strategy}/fold_{fold}")
    metrics  = _run_inference_and_eval(model, cfg, strategy, fold, device, save_dir)
    metrics['training_history'] = history

    torch.save(model.state_dict(), os.path.join(save_dir, "finetuned_model.pth"))
    print(f"  Dice={metrics['dice']:.4f}  Prec={metrics['precision']:.4f}  "
          f"Rec={metrics['recall']:.4f}  clDice={metrics.get('cldice','N/A')}")
    return metrics


def run_experiment_c(cfg, strategy, fold, device):
    print(f"\n{'='*60}")
    print(f"EXP C: Train From Scratch | strategy={strategy} | fold={fold}")
    print(f"{'='*60}")

    model = create_fresh_model(device)

    train_loader, _ = create_dataloaders(
        cfg.data.processed_dir, strategy, fold, cfg.training)

    loss_fn   = CombinedLoss(dice_weight=0.5, bce_weight=0.5, pos_weight=9.0)
    optimizer = AdamW(model.parameters(),
                      lr=cfg.training.scratch_lr,
                      weight_decay=cfg.training.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.training.scratch_epochs)

    history  = train_model(model, train_loader, loss_fn, optimizer, scheduler,
                           cfg.training.scratch_epochs, device,
                           cfg.training.patience, "from-scratch")

    save_dir = os.path.join(cfg.results_dir, f"exp_c/{strategy}/fold_{fold}")
    metrics  = _run_inference_and_eval(model, cfg, strategy, fold, device, save_dir)
    metrics['training_history'] = history

    print(f"  Dice={metrics['dice']:.4f}  Prec={metrics['precision']:.4f}  "
          f"Rec={metrics['recall']:.4f}  clDice={metrics.get('cldice','N/A')}")
    return metrics


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir",    default="./Extend_data/vessap_processed")
    parser.add_argument("--pretrained_model", default="../runs_final/fold0/checkpoints/best.pth")
    parser.add_argument("--output_dir",       default="./results")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--strategies", nargs='+', default=["ch0_only", "max_fusion"])
    parser.add_argument("--skip_a", action="store_true")
    parser.add_argument("--skip_b", action="store_true")
    parser.add_argument("--skip_c", action="store_true")
    args = parser.parse_args()

    cfg = ExperimentConfig()
    cfg.data.processed_dir         = args.processed_dir
    cfg.data.pretrained_model_path = args.pretrained_model
    cfg.results_dir                = args.output_dir
    cfg.data.channel_strategies    = args.strategies

    device = torch.device(args.device)
    print(f"Device: {device}")

    torch.manual_seed(cfg.training.seed)
    np.random.seed(cfg.training.seed)

    all_results = {}

    for strategy in cfg.data.channel_strategies:
        for fold in [1, 2]:
            key = f"{strategy}/fold_{fold}"
            if not args.skip_a:
                all_results[f"exp_a/{key}"] = run_experiment_a(cfg, strategy, fold, device)
            if not args.skip_b:
                all_results[f"exp_b/{key}"] = run_experiment_b(cfg, strategy, fold, device)
            if not args.skip_c:
                all_results[f"exp_c/{key}"] = run_experiment_c(cfg, strategy, fold, device)

    # Serialize and save
    def _serial(obj):
        if isinstance(obj, (np.floating, np.integer)): return obj.item()
        if isinstance(obj, np.ndarray):                return obj.tolist()
        if isinstance(obj, dict):  return {k: _serial(v) for k, v in obj.items()}
        if isinstance(obj, list):  return [_serial(v) for v in obj]
        return obj

    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, "all_results.json")
    with open(results_path, 'w') as f:
        json.dump(_serial(all_results), f, indent=2)

    # Summary table
    print(f"\n{'='*75}")
    print(f"{'EXPERIMENT SUMMARY':^75}")
    print(f"{'='*75}")
    print(f"{'Experiment':<35} {'Dice':>8} {'Prec':>8} {'Recall':>8} {'clDice':>8}")
    print(f"{'-'*75}")
    for key in sorted(all_results.keys()):
        r      = all_results[key]
        cldice = r.get('cldice', float('nan'))
        if isinstance(cldice, str): cldice = float('nan')
        print(f"{key:<35} {r['dice']:>8.4f} {r['precision']:>8.4f} "
              f"{r['recall']:>8.4f} {cldice:>8.4f}")
    print(f"{'='*75}")
    print(f"\nSaved → {results_path}")


if __name__ == "__main__":
    main()
