"""
config.py - Centralized configuration for VesSAP cross-domain fine-tuning experiments.
"""
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import os


@dataclass
class DataConfig:
    """Data paths and preprocessing settings."""

    # === PATHS ===
    vessap_data_dir: str = "./Extend_data/extend_raw"       # ch0 and ch1 images
    vessap_label_dir: str = "./Extend_data/extend_seg"       # label files
    processed_dir: str = "./Extend_data/vessap_processed"    # output for preprocessed data
    pretrained_model_path: str = "../runs_final/fold0/checkpoints/best.pth"

    # === SAMPLE DEFINITIONS ===
    # (ch0_filename, ch1_filename, label_filename)
    vessap_samples: List[Tuple[str, str, str]] = field(default_factory=lambda: [
        ("image_1_ch0.nii.gz", "image_1_ch1.nii.gz", "label_1.nii.gz"),
        ("image_2_ch0.nii.gz", "image_2_ch1.nii.gz", "label_2.nii.gz"),
    ])

    # === CHANNEL FUSION ===
    # "ch0_only", "ch1_only", "max_fusion", "mean_fusion"
    channel_strategies: List[str] = field(default_factory=lambda: ["ch0_only", "max_fusion"])

    # === INTENSITY NORMALIZATION ===
    clip_percentile_low: float = 0.5
    clip_percentile_high: float = 99.5
    use_histogram_matching: bool = False
    minivess_mean: float = 0.15
    minivess_std: float = 0.12

    # === PATCH EXTRACTION ===
    patch_size: Tuple[int, int, int] = (64, 64, 64)
    patch_overlap: float = 0.25
    min_vessel_ratio: float = 0.001


@dataclass
class TrainingConfig:
    """Training/fine-tuning hyperparameters."""

    # Fine-tuning (Experiment B)
    finetune_epochs: int = 50
    finetune_lr: float = 1e-4
    finetune_lr_scheduler: str = "cosine"
    finetune_freeze_encoder: bool = False
    finetune_freeze_epochs: int = 10

    # Training from scratch (Experiment C)
    scratch_epochs: int = 100
    scratch_lr: float = 1e-3

    # Common
    batch_size: int = 4
    num_workers: int = 4
    optimizer: str = "adamw"
    weight_decay: float = 1e-5
    loss_function: str = "dice_ce"
    dice_loss_weight: float = 0.5
    ce_loss_weight: float = 0.5

    # Augmentation
    use_augmentation: bool = True
    aug_flip_prob: float = 0.5
    aug_rotate_prob: float = 0.3
    aug_intensity_shift: float = 0.1
    aug_intensity_scale: float = 0.1

    # Early stopping
    patience: int = 15
    min_delta: float = 0.001

    seed: int = 42


@dataclass
class EvalConfig:
    """Evaluation settings."""
    compute_dice: bool = True
    compute_cldice: bool = True
    compute_precision_recall: bool = True
    compute_hausdorff: bool = False

    sliding_window_batch_size: int = 8
    inference_overlap: float = 0.25

    threshold: float = 0.5
    search_threshold: bool = True
    threshold_range: Tuple[float, float, float] = (0.1, 0.9, 0.05)


@dataclass
class ExperimentConfig:
    """Full experiment configuration."""
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    results_dir: str = "./results"
    figures_dir: str = "./figures"
    log_dir: str = "./logs"

    run_exp_a: bool = True
    run_exp_b: bool = True
    run_exp_c: bool = True

    def __post_init__(self):
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.figures_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.data.processed_dir, exist_ok=True)
