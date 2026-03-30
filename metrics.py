"""
metrics.py — Evaluation metrics for binary vessel segmentation
===============================================================
All metrics work on raw logits or probabilities, handling threshold internally.
Designed to be reused across all model variants (baseline U-Net, 2.5D, etc.)

Metrics implemented:
    - Dice coefficient        (primary metric)
    - IoU / Jaccard index
    - Hausdorff Distance 95   (HD95, boundary quality)
    - Pixel accuracy
    - Precision / Recall / F1

Usage:
    from metrics import SegmentationMetrics

    evaluator = SegmentationMetrics(threshold=0.5)

    # Per-batch update (during val loop)
    evaluator.update(logits, labels)

    # End of epoch
    results = evaluator.compute()
    evaluator.reset()
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt


# ─────────────────────────────────────────────────────────────────────────────
# Individual metric functions (numpy, operate on binary 2D arrays)
# ─────────────────────────────────────────────────────────────────────────────

def dice_score(pred, target, smooth=1e-6):
    """Dice = 2*|P∩T| / (|P|+|T|)"""
    inter = (pred * target).sum()
    return (2.0 * inter + smooth) / (pred.sum() + target.sum() + smooth)


def iou_score(pred, target, smooth=1e-6):
    """IoU = |P∩T| / |P∪T|"""
    inter = (pred * target).sum()
    union = pred.sum() + target.sum() - inter
    return (inter + smooth) / (union + smooth)


def hausdorff_distance_95(pred, target):
    """
    95th percentile Hausdorff Distance between prediction and target boundaries.
    Returns 0.0 if either mask is empty (no vessel in that slice).
    """
    if pred.sum() == 0 or target.sum() == 0:
        return 0.0

    # Distance transform from the surface of each mask
    pred_dt   = distance_transform_edt(~pred.astype(bool))
    target_dt = distance_transform_edt(~target.astype(bool))

    # Surface points (boundary pixels of each mask)
    pred_surf   = pred.astype(bool) & ~_erode(pred.astype(bool))
    target_surf = target.astype(bool) & ~_erode(target.astype(bool))

    # Distances from one surface to the other
    dist_pred_to_target   = pred_dt[target_surf]
    dist_target_to_pred   = target_dt[pred_surf]

    if len(dist_pred_to_target) == 0 or len(dist_target_to_pred) == 0:
        return 0.0

    hd95 = max(
        np.percentile(dist_pred_to_target,   95),
        np.percentile(dist_target_to_pred, 95),
    )
    return float(hd95)


def _erode(binary_mask):
    """Simple 3x3 erosion for boundary extraction."""
    from scipy.ndimage import binary_erosion
    return binary_erosion(binary_mask, iterations=1)


def precision_recall_f1(pred, target, smooth=1e-6):
    tp = (pred * target).sum()
    fp = pred.sum() - tp
    fn = target.sum() - tp
    precision = (tp + smooth) / (tp + fp + smooth)
    recall    = (tp + smooth) / (tp + fn + smooth)
    f1        = 2 * precision * recall / (precision + recall + smooth)
    return float(precision), float(recall), float(f1)


def pixel_accuracy(pred, target):
    return float((pred == target).mean())


# ─────────────────────────────────────────────────────────────────────────────
# Accumulator — use during validation loop
# ─────────────────────────────────────────────────────────────────────────────

class SegmentationMetrics:
    """
    Accumulates per-slice metrics across a full validation epoch,
    then returns mean values.

    Parameters
    ----------
    threshold   : float  — sigmoid threshold for binarisation (default 0.5)
    compute_hd95: bool   — HD95 is slow; disable for quick sanity checks
    """

    def __init__(self, threshold=0.5, compute_hd95=True):
        self.threshold    = threshold
        self.compute_hd95 = compute_hd95
        self.reset()

    def reset(self):
        self._dice      = []
        self._iou       = []
        self._hd95      = []
        self._acc       = []
        self._precision = []
        self._recall    = []
        self._f1        = []

    def update(self, logits, labels):
        """
        Parameters
        ----------
        logits : torch.Tensor (B, 1, H, W)  — raw model output (before sigmoid)
        labels : torch.Tensor (B, 1, H, W)  — binary ground truth {0, 1}
        """
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        lbls  = labels.detach().cpu().numpy()

        for b in range(probs.shape[0]):
            pred   = (probs[b, 0] >= self.threshold).astype(np.float32)
            target = (lbls[b, 0] > 0).astype(np.float32)

            self._dice.append(float(dice_score(pred, target)))
            self._iou.append(float(iou_score(pred, target)))
            self._acc.append(pixel_accuracy(pred, target))

            p, r, f = precision_recall_f1(pred, target)
            self._precision.append(p)
            self._recall.append(r)
            self._f1.append(f)

            if self.compute_hd95:
                self._hd95.append(hausdorff_distance_95(pred, target))

    def compute(self):
        """Return dict of mean metrics over all accumulated samples."""
        results = {
            'dice':      float(np.mean(self._dice)),
            'iou':       float(np.mean(self._iou)),
            'accuracy':  float(np.mean(self._acc)),
            'precision': float(np.mean(self._precision)),
            'recall':    float(np.mean(self._recall)),
            'f1':        float(np.mean(self._f1)),
        }
        if self.compute_hd95 and self._hd95:
            results['hd95'] = float(np.mean(self._hd95))
        return results

    def compute_and_print(self, prefix=''):
        results = self.compute()
        header = f'{prefix} ' if prefix else ''
        print(f'{header}Dice={results["dice"]:.4f}  '
              f'IoU={results["iou"]:.4f}  '
              f'HD95={results.get("hd95", float("nan")):.2f}  '
              f'Prec={results["precision"]:.4f}  '
              f'Rec={results["recall"]:.4f}')
        return results


# ─────────────────────────────────────────────────────────────────────────────
# Loss functions
# ─────────────────────────────────────────────────────────────────────────────

class DiceLoss(nn.Module):
    """Differentiable Dice loss (1 - Dice)."""

    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        inter = (probs * targets).sum(dim=(2, 3))
        denom = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice  = (2.0 * inter + self.smooth) / (denom + self.smooth)
        return 1.0 - dice.mean()


class CombinedLoss(nn.Module):
    """
    Dice + BCE loss — standard for sparse vessel segmentation.
    loss = dice_weight * Dice + bce_weight * BCE
    pos_weight: upweight vessel class (e.g. 9.0 if vessels are ~10% of pixels)
    """

    def __init__(self, dice_weight=0.5, bce_weight=0.5, pos_weight=None):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight  = bce_weight
        self.dice_loss   = DiceLoss()
        self.pos_weight  = pos_weight

    def forward(self, logits, targets):
        if self.pos_weight is not None:
            pw  = torch.tensor([self.pos_weight], device=logits.device)
            bce = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, targets, pos_weight=pw)
        else:
            bce = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, targets)
        return self.dice_weight * self.dice_loss(logits, targets) + self.bce_weight * bce


# ─────────────────────────────────────────────────────────────────────────────
# Quick check
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    B, H, W = 4, 128, 128
    logits = torch.randn(B, 1, H, W)
    labels = (torch.rand(B, 1, H, W) > 0.9).float()

    evaluator = SegmentationMetrics(compute_hd95=True)
    evaluator.update(logits, labels)
    results = evaluator.compute_and_print(prefix='[test]')
    print(results)

    loss_fn = CombinedLoss(pos_weight=9.0)
    loss = loss_fn(logits, labels)
    print(f'CombinedLoss: {loss.item():.4f}')
