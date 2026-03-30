"""
metrics.py - Evaluation metrics for vessel segmentation.

Includes: Dice, clDice (centerline Dice), Precision, Recall, Hausdorff95.
"""

import numpy as np
from typing import Dict, Optional
from scipy import ndimage


def dice_score(pred: np.ndarray, target: np.ndarray) -> float:
    """Standard Dice coefficient (F1 score for binary segmentation)."""
    pred = pred.astype(bool)
    target = target.astype(bool)
    intersection = np.logical_and(pred, target).sum()
    if pred.sum() + target.sum() == 0:
        return 1.0  # Both empty → perfect match
    return 2.0 * intersection / (pred.sum() + target.sum())


def precision_score(pred: np.ndarray, target: np.ndarray) -> float:
    """Precision: TP / (TP + FP)."""
    pred = pred.astype(bool)
    target = target.astype(bool)
    tp = np.logical_and(pred, target).sum()
    if pred.sum() == 0:
        return 1.0 if target.sum() == 0 else 0.0
    return tp / pred.sum()


def recall_score(pred: np.ndarray, target: np.ndarray) -> float:
    """Recall (sensitivity): TP / (TP + FN)."""
    pred = pred.astype(bool)
    target = target.astype(bool)
    tp = np.logical_and(pred, target).sum()
    if target.sum() == 0:
        return 1.0 if pred.sum() == 0 else 0.0
    return tp / target.sum()


# ============================================================================
# CENTERLINE DICE (clDice) - Topology-aware metric
# ============================================================================

def _skeletonize_3d(binary_volume: np.ndarray) -> np.ndarray:
    """
    3D skeletonization using scipy morphological operations.
    For better results, use skimage.morphology.skeletonize_3d if available.
    """
    try:
        from skimage.morphology import skeletonize_3d
        return skeletonize_3d(binary_volume.astype(np.uint8)).astype(bool)
    except ImportError:
        # Fallback: simple thinning via distance transform
        # Not as accurate but avoids dependency
        from scipy.ndimage import distance_transform_edt
        dist = distance_transform_edt(binary_volume)
        # Local maxima of distance transform approximate the skeleton
        from scipy.ndimage import maximum_filter
        local_max = maximum_filter(dist, size=3) == dist
        skeleton = np.logical_and(local_max, binary_volume)
        return skeleton


def cl_dice(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Centerline Dice (clDice): topology-preserving metric for tubular structures.
    
    clDice = 2 * Tprec * Tsens / (Tprec + Tsens)
    where:
        Tprec = |S(pred) ∩ target| / |S(pred)|    (topology precision)
        Tsens = |S(target) ∩ pred| / |S(target)|  (topology sensitivity)
        S(·) = skeletonization
    
    Reference: Shit et al., "clDice - a Novel Topology-Preserving Loss Function
               for Tubular Structure Segmentation", CVPR 2021.
    """
    pred = pred.astype(bool)
    target = target.astype(bool)

    if pred.sum() == 0 and target.sum() == 0:
        return 1.0
    if pred.sum() == 0 or target.sum() == 0:
        return 0.0

    # Skeletonize both
    skel_pred = _skeletonize_3d(pred)
    skel_target = _skeletonize_3d(target)

    if skel_pred.sum() == 0 or skel_target.sum() == 0:
        return 0.0

    # Topology precision: how much of pred's skeleton falls within the target
    tprec = np.logical_and(skel_pred, target).sum() / skel_pred.sum()
    # Topology sensitivity: how much of target's skeleton falls within the pred
    tsens = np.logical_and(skel_target, pred).sum() / skel_target.sum()

    if tprec + tsens == 0:
        return 0.0

    return 2.0 * tprec * tsens / (tprec + tsens)


# ============================================================================
# HAUSDORFF DISTANCE (optional, can be slow)
# ============================================================================

def hausdorff_95(pred: np.ndarray, target: np.ndarray) -> float:
    """
    95th percentile Hausdorff distance.
    Returns distance in voxel units.
    """
    from scipy.ndimage import distance_transform_edt

    pred = pred.astype(bool)
    target = target.astype(bool)

    if pred.sum() == 0 or target.sum() == 0:
        return float('inf')

    # Distance from each pred surface voxel to nearest target surface voxel
    dist_pred_to_target = distance_transform_edt(~target)
    dist_target_to_pred = distance_transform_edt(~pred)

    # Surface voxels (boundary)
    from scipy.ndimage import binary_erosion
    pred_surface = np.logical_xor(pred, binary_erosion(pred))
    target_surface = np.logical_xor(target, binary_erosion(target))

    if pred_surface.sum() == 0 or target_surface.sum() == 0:
        return float('inf')

    d_p2t = dist_pred_to_target[pred_surface]
    d_t2p = dist_target_to_pred[target_surface]

    hd95 = max(np.percentile(d_p2t, 95), np.percentile(d_t2p, 95))
    return float(hd95)


# ============================================================================
# UNIFIED EVALUATION
# ============================================================================

def compute_all_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    compute_cldice: bool = True,
    compute_hd95: bool = False,
) -> Dict[str, float]:
    """
    Compute all metrics for a single volume prediction.
    
    Args:
        pred: Binary prediction, shape (D, H, W)
        target: Binary ground truth, shape (D, H, W)
        compute_cldice: Whether to compute clDice (slower)
        compute_hd95: Whether to compute Hausdorff distance (slowest)
    
    Returns:
        Dictionary of metric_name -> value
    """
    results = {
        'dice': dice_score(pred, target),
        'precision': precision_score(pred, target),
        'recall': recall_score(pred, target),
        'pred_fg_ratio': float(pred.astype(bool).sum() / pred.size),
        'target_fg_ratio': float(target.astype(bool).sum() / target.size),
    }

    if compute_cldice:
        try:
            results['cldice'] = cl_dice(pred, target)
        except Exception as e:
            print(f"  [Warning] clDice computation failed: {e}")
            results['cldice'] = float('nan')

    if compute_hd95:
        try:
            results['hausdorff_95'] = hausdorff_95(pred, target)
        except Exception as e:
            print(f"  [Warning] HD95 computation failed: {e}")
            results['hausdorff_95'] = float('nan')

    return results


def find_optimal_threshold(
    pred_prob: np.ndarray,
    target: np.ndarray,
    threshold_range: tuple = (0.1, 0.9, 0.05),
) -> tuple:
    """
    Search for the threshold that maximizes Dice score.
    
    Returns:
        (best_threshold, best_dice, all_results)
    """
    start, end, step = threshold_range
    thresholds = np.arange(start, end + step, step)

    best_dice = -1
    best_thresh = 0.5
    all_results = []

    for t in thresholds:
        pred_binary = (pred_prob > t).astype(bool)
        d = dice_score(pred_binary, target)
        all_results.append({'threshold': float(t), 'dice': float(d)})
        if d > best_dice:
            best_dice = d
            best_thresh = float(t)

    return best_thresh, best_dice, all_results
