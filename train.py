"""
train.py — Training script for baseline U-Net vessel segmentation
==================================================================
Usage:
    # 单折训练
    python train.py --npz_dir ./data/processed --fold 0

    # 全部5折
    python train.py --npz_dir ./data/processed --run_all_folds

    # 自定义参数
    python train.py --npz_dir ./data/processed --fold 0 \
                    --epochs 100 --batch_size 16 --lr 1e-4 --window_k 1
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset import get_dataloaders
from unet import UNet
from metrics import SegmentationMetrics, CombinedLoss
from visualize_results import save_prediction_figure, save_learning_curves
from interslice_unet import InterSliceUNet
from full_attention_unet import FullAttentionUNet
from attentionGate_unet import AttentionUNet

# ─────────────────────────────────────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser()

    # Data
    p.add_argument('--npz_dir',       type=str,   required=True)
    p.add_argument('--out_dir',       type=str,   default='./runs')

    # Cross-validation
    p.add_argument('--fold',          type=int,   default=0)
    p.add_argument('--n_folds',       type=int,   default=5)
    p.add_argument('--run_all_folds', action='store_true')

    # Model
    p.add_argument('--window_k',      type=int,   default=1,
                   help='2.5D half-width: k=1→3ch, k=2→5ch, k=0→2D')
    p.add_argument('--base_channels', type=int,   default=64)

    # Training
    p.add_argument('--epochs',        type=int,   default=100)
    p.add_argument('--batch_size',    type=int,   default=16)
    p.add_argument('--lr',            type=float, default=1e-4)
    p.add_argument('--weight_decay',  type=float, default=1e-5)
    p.add_argument('--pos_weight',    type=float, default=9.0)
    p.add_argument('--dice_weight',   type=float, default=0.5)
    p.add_argument('--bce_weight',    type=float, default=0.5)
    p.add_argument('--use_all', action='store_true')

    # Misc
    p.add_argument('--num_workers',   type=int,   default=4)
    p.add_argument('--seed',          type=int,   default=42)
    p.add_argument('--viz_every',     type=int,   default=10)
    p.add_argument('--device',        type=str,   default='cuda')
    p.add_argument('--crop_size', type=int, default=None)
    
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# One epoch
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss   = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


def val_epoch(model, loader, loss_fn, evaluator, device):
    model.eval()
    total_loss = 0.0
    evaluator.reset()

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss   = loss_fn(logits, labels)
            total_loss += loss.item() * images.size(0)

            evaluator.update(logits, labels)

    return total_loss / len(loader.dataset)


# ─────────────────────────────────────────────────────────────────────────────
# Train one fold
# ─────────────────────────────────────────────────────────────────────────────

def train_fold(args, fold):
    torch.manual_seed(args.seed + fold)
    np.random.seed(args.seed + fold)

    # Output dirs
    run_dir  = Path(args.out_dir) / f'fold{fold}'
    ckpt_dir = run_dir / 'checkpoints'
    fig_dir  = run_dir / 'figures'
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(exist_ok=True)
    fig_dir.mkdir(exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'\n{"="*60}')
    print(f'Fold {fold}/{args.n_folds}  |  device: {device}')
    print(f'{"="*60}')

    # Data
    train_loader, val_loader = get_dataloaders(
        npz_dir    = args.npz_dir,
        fold       = fold,
        n_folds    = args.n_folds,
        batch_size = args.batch_size,
        window_k   = args.window_k,
        num_workers= args.num_workers,
        seed       = args.seed,
        crop_size  = args.crop_size
    )

    # Model _Unet
    # in_channels = 2 * args.window_k + 1
    # model = UNet(
    #     in_channels   = in_channels,
    #     out_channels  = 1,
    #     base_channels = args.base_channels,
    # ).to(device)
    # print(f'Model: UNet  in_ch={in_channels}  '
    #       f'base_ch={args.base_channels}  '
    #       f'params={model.count_parameters():,}')

    # Model _InterSliceUNet
    # in_channels = 2 * args.window_k + 1
    # model = InterSliceUNet(
    #     in_channels   = in_channels,
    #     out_channels  = 1,
    #     base_channels = args.base_channels,
    # ).to(device)
    # print(f'Model: InterSliceUNet  in_ch={in_channels}  '
    #       f'base_ch={args.base_channels}  '
    #       f'params={model.count_parameters():,}')

    # Model _AttentionGateUNet
    # in_channels = 2 * args.window_k + 1
    # model = AttentionGateUNet(
    #     in_channels   = in_channels,
    #     out_channels  = 1,
    #     base_channels = args.base_channels,
    # ).to(device)
    # print(f'Model: AttentionGateUNet  in_ch={in_channels}  '
    #       f'base_ch={args.base_channels}  '
    #       f'params={model.count_parameters():,}')

    # Model _Full_Attention_UNet
    in_channels = 2 * args.window_k + 1
    model = FullAttentionUNet(
        in_channels   = in_channels,
        out_channels  = 1,
        base_channels = args.base_channels,
    ).to(device)
    print(f'Model: Full_Attention_UNet  in_ch={in_channels}  '
          f'base_ch={args.base_channels}  '
          f'params={model.count_parameters():,}')
          
    # Loss, optimiser, scheduler
    loss_fn   = CombinedLoss(
        dice_weight = args.dice_weight,
        bce_weight  = args.bce_weight,
        pos_weight  = args.pos_weight,
    )

    optimizer = optim.AdamW(
    model.parameters(),
    lr           = args.lr,
    weight_decay = args.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    evaluator = SegmentationMetrics(compute_hd95=False)   # HD95 slow, enable at end

    # Resume from checkpoint if exists
    start_epoch = 1
    latest_ckpt = ckpt_dir / "latest.pth"
    if latest_ckpt.exists():
        ckpt = torch.load(latest_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])      
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])      
        train_losses = ckpt.get("train_losses", [])
        val_losses   = ckpt.get("val_losses", [])
        val_dices    = ckpt.get("val_dices", [])
        best_dice    = max(val_dices) if val_dices else 0.0
        start_epoch  = ckpt["epoch"] + 1
        print(f" Resumed from epoch {ckpt['epoch']}")

    # Training loop
    best_dice      = 0.0
    train_losses   = []
    val_losses     = []
    val_dices      = []

    # Save config
    config = vars(args)
    config['fold'] = fold
    config['in_channels'] = in_channels
    with open(run_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss   = val_epoch(model, val_loader, loss_fn, evaluator, device)
        metrics    = evaluator.compute()
        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_dices.append(metrics['dice'])

        elapsed = time.time() - t0
        print(f'Epoch {epoch:3d}/{args.epochs}  '
              f'train_loss={train_loss:.4f}  '
              f'val_loss={val_loss:.4f}  '
              f'dice={metrics["dice"]:.4f}  '
              f'iou={metrics["iou"]:.4f}  '
              f'lr={scheduler.get_last_lr()[0]:.2e}  '
              f't={elapsed:.1f}s')

        # Save best checkpoint
        if metrics['dice'] > best_dice:
            best_dice = metrics['dice']
            torch.save({
                'epoch':             epoch,
                'fold':              fold,
                'model_state_dict':  model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_dice':         best_dice,
                'metrics':           metrics,
                'in_channels':       in_channels,
                'base_channels':     args.base_channels,
            }, ckpt_dir / 'best.pth')
            print(f'  ✓ Best model saved  (dice={best_dice:.4f})')

        # Save latest checkpoint (for resuming)
        torch.save({
            'epoch':             epoch,
            'model_state_dict':  model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_losses':      train_losses,
            'val_losses':        val_losses,
            'val_dices':         val_dices,
            'in_channels':       in_channels,
        }, ckpt_dir / 'latest.pth')

        # Visualise predictions periodically
        if epoch % args.viz_every == 0 or epoch == args.epochs:
            model.eval()
            images_b, labels_b = next(iter(val_loader))
            with torch.no_grad():
                logits_b = model(images_b.to(device)).cpu()
            save_prediction_figure(
                images   = images_b.numpy()[:8],
                labels   = labels_b.numpy()[:8],
                preds    = logits_b.numpy()[:8],
                out_path = fig_dir / f'epoch_{epoch:03d}.png',
                title    = f'Fold {fold} — Epoch {epoch}  Dice={metrics["dice"]:.4f}',
            )

    # Final learning curves
    save_learning_curves(
        train_losses = train_losses,
        val_losses   = val_losses,
        val_dices    = val_dices,
        out_path     = run_dir / 'learning_curves.png',
    )

    # Final evaluation with HD95
    print(f'\nFold {fold} — Final evaluation (with HD95)...')
    evaluator_full = SegmentationMetrics(compute_hd95=True)
    val_epoch(model, val_loader, loss_fn, evaluator_full, device)
    final_metrics = evaluator_full.compute_and_print(prefix=f'[Fold {fold}]')

    # Save fold summary
    with open(run_dir / 'results.json', 'w') as f:
        json.dump({
            'fold':          fold,
            'best_dice':     best_dice,
            'final_metrics': final_metrics,
        }, f, indent=2)

    print(f'\nFold {fold} done. Best dice: {best_dice:.4f}')
    return best_dice, final_metrics


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = get_args()

    if args.run_all_folds:
        all_metrics = []
        for fold in range(args.n_folds):
            best_dice, metrics = train_fold(args, fold)
            all_metrics.append(metrics)

        # Cross-validation summary
        print(f'\n{"="*60}')
        print(f'Cross-validation summary ({args.n_folds} folds)')
        print(f'{"="*60}')
        for key in ['dice', 'iou', 'precision', 'recall']:
            vals = [m[key] for m in all_metrics]
            print(f'  {key:12s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}')
        if 'hd95' in all_metrics[0]:
            vals = [m['hd95'] for m in all_metrics]
            print(f'  {"hd95":12s}: {np.mean(vals):.2f} ± {np.std(vals):.2f}')

        # Save CV summary
        summary = {
            'n_folds': args.n_folds,
            'per_fold': all_metrics,
            'mean': {k: float(np.mean([m[k] for m in all_metrics]))
                     for k in all_metrics[0]},
            'std':  {k: float(np.std([m[k] for m in all_metrics]))
                     for k in all_metrics[0]},
        }
        with open(Path(args.out_dir) / 'cv_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

    else:
        train_fold(args, args.fold)


if __name__ == '__main__':
    main()
