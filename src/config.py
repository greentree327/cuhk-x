"""
Configuration dataclass for the CUHK-X Small Model Track pipeline.
Centralizes all hyperparameters, paths, and training settings.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Project root
ROOT = Path(__file__).resolve().parent.parent


@dataclass
class FeatureFlags:
    """Toggleable feature flags for ablation studies.

    Set any to False to disable that technique and isolate its impact.
    All defaults = True (best known configuration from CMI + EDA).
    """
    # -- Feature engineering --
    use_synthesized_features: bool = False     # master toggle: True = full CMI features

    # -- Background invariance techniques --
    use_skeleton_attention_mask: bool = True   # skeleton-guided spatial attention on frames
    use_spatial_crop: bool = True              # random crop+resize on frame modalities
    use_random_erase: bool = True              # RandomErase rectangular patches on frames
    erase_scale: tuple = (0.02, 0.10)          # erase area range as fraction of frame
    erase_ratio: tuple = (0.3, 3.3)            # erase aspect ratio range
    spatial_crop_scale: tuple = (0.75, 1.0)    # crop size range as fraction of frame
    mask_bg_weight: float = 0.15               # background attention weight (vs 1.0 foreground)

    # -- Temporal techniques --
    use_segment_pooling: bool = False          # 3-segment (early/mid/late) temporal pooling for time-series
    segment_count: int = 3                     # number of segments for pooling

    # -- Auxiliary supervision --
    use_aux_category_loss: bool = True         # coarse 8-category auxiliary classifier
    aux_loss_weight: float = 0.2               # weight of auxiliary loss in total
    num_categories: int = 8                    # coarse action groups

    # -- Other --
    use_class_weights: bool = True             # inverse-sqrt class weighting


ACTION_CATEGORIES = {
    0: 0, 1: 0, 2: 0,                                          # grooming: wash face, brush teeth, comb hair
    3: 1, 5: 1, 16: 1,                                          # dressing: take off/put on/fold clothes
    6: 2, 7: 2, 8: 2, 9: 2, 10: 2, 11: 2,                       # eating: drink, eat, tableware, pour, stir, peel
    12: 3, 13: 3, 14: 3, 15: 3,                                  # cleaning: sweep, mop, wipe bowls, wipe windows
    17: 4, 18: 4, 19: 4, 20: 4, 21: 4, 22: 4, 23: 4, 24: 4, 25: 4, 26: 4, 27: 4,  # tech: keyboard, write, call, check time, read, turn pages, music, mobile, watch TV, games, selfie
    28: 5, 29: 5, 30: 5, 31: 5, 35: 5, 36: 5,                    # exercise: jog, squats, jumping jacks, stretch, lunges, walk
    32: 6, 33: 6, 34: 6,                                          # posture: stand up, lie down, sit down
    4: 7, 37: 7, 38: 7, 39: 7,                                    # health: wipe hands, take medicine, massage, body temp
}


@dataclass
class Config:
    """Master configuration for the CUHK-X HAR pipeline."""

    # -- Feature flags (toggle for ablation studies) --
    flags: FeatureFlags = field(default_factory=FeatureFlags)

    # -- Paths (updated for actual extracted data location) --
    data_root: Path = ROOT / "CUHK-X_Small_Model_Track" / "Small-Model-Track"
    train_data: Path = ROOT / "CUHK-X_Small_Model_Track" / "Small-Model-Track" / "Training" / "data" / "HAR_extracted" / "HAR" / "data"
    test_data: Path = ROOT / "CUHK-X_Small_Model_Track" / "Small-Model-Track" / "Testing" / "data" / "small_model_track_test" / "small_model_track_test"
    test_csv: Path = ROOT / "CUHK-X_Small_Model_Track" / "Small-Model-Track" / "Testing" / "test_file" / "test.csv"
    class_mapping: Path = ROOT / "CUHK-X_Small_Model_Track" / "Small-Model-Track" / "class_mapping.csv"
    output_dir: Path = ROOT / "output"

    # -- Data --
    modalities: tuple = ("Depth_Color", "IR", "Thermal", "IMU", "Radar", "Skeleton")
    num_classes: int = 40

    # Frame modality settings
    frame_size: int = 160          # resize frames to frame_size × frame_size
    num_frames: int = 16           # uniform sample N frames for Depth/IR/Skel (10fps)
    num_thermal_frames: int = 32   # Thermal has 118 frames at 25fps; sample more
    frame_base_width: int = 32     # base channels for custom lightweight CNN
    depth_frames_native: int = 42  # Depth_Color ~42 frames at ~10fps
    ir_frames_native: int = 42     # IR ~42 frames at ~10fps
    thermal_frames_native: int = 118  # Thermal ~118 frames at ~25fps

    # IMU settings (5 WitMotion sensors: LL, RL, LA, RA, Chest)
    imu_target_hz: int = 20
    imu_seq_len: int = 128

    # Radar settings (TI IWR6843ISK, 60-64 GHz, ~20 fps)
    radar_max_points: int = 64
    radar_point_dim: int = 6       # x, y, z, v(doppler), snr, noise
    radar_seq_len: int = 82

    # Skeleton settings
    skel_num_joints: int = 17
    skel_joint_dim: int = 3        # x, y, z per joint
    skel_seq_len: int = 42

    # -- Model --
    encoder_dim: int = 256         # output dim of each modality encoder
    fusion_hidden: int = 512       # hidden dim in fusion MLP
    num_categories: int = 8        # coarse action categories for auxiliary head
    dropout: float = 0.3           # dropout rate
    weight_decay: float = 1e-4     # L2 regularization for SE-CNN blocks

    # -- Cross-subject split (HARDCODED per competition rules) --
    # Train: users 1–9, 16–24 | Test: users 10–11, 25–26
    train_users: tuple = tuple(range(1, 10)) + tuple(range(16, 25))  # 1-9, 16-24
    val_users: tuple = ()        # if empty, use subset of train_users for val
    val_ratio: float = 0.15      # fraction of train_users to hold out for validation
    test_users: tuple = (10, 11, 25, 26)

    # -- Training --
    seed: int = 42
    n_folds: int = 5
    batch_size: int = 32
    epochs: int = 100
    early_stop_patience: int = 20
    lr: float = 1e-3
    lr_min: float = 1e-6
    warmup_epochs: int = 5
    label_smoothing: float = 0.1
    # EMA time constant is 1/(1-decay) steps. 0.999 (~1000 steps) was ported
    # from CMI, whose dataset had far more steps/epoch than this one's
    # ~37-39 (599-622 train clips / batch_size 16). At 0.999, validation
    # (always evaluated on EMA weights) needs ~18-27 epochs just to start
    # reflecting real progress — confirmed live: train acc/loss improved
    # normally epoch-to-epoch while val stayed flat, because EMA hadn't
    # caught up yet. That's within early_stop_patience=20, so early
    # stopping could trigger before EMA-based validation reflects the
    # model at all. 0.98 (~50-step / ~1.3-epoch time constant) still
    # smooths noisy updates but converges fast enough to be meaningful
    # within a handful of epochs.
    ema_decay: float = 0.98
    grad_clip_norm: float = 1.0
    mixed_precision: bool = True

    # -- Augmentation --
    mixup_alpha: float = 0.3
    time_shift_pct: float = 0.25     # max shift fraction
    time_stretch_range: tuple = (0.8, 1.2)
    noise_std: float = 0.01
    modality_dropout_p: float = 0.2  # reduced: only 1.9% truly missing; simulate more for robustness
    frame_dropout_p: float = 0.1

    # -- Inference --
    tta: bool = True               # test-time augmentation
    tta_crops: int = 5             # number of temporal crops for TTA

    # -- Compute --
    device: str = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    num_workers: int = 4

    def __post_init__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def imu_input_dim(self):
        """IMU feature dimension: 50 raw, 135 synthesized."""
        return 135 if self.flags.use_synthesized_features else 50

    @property
    def skel_input_dim(self):
        """Skeleton feature dimension: 51 raw, 119 synthesized."""
        return 119 if self.flags.use_synthesized_features else 51


# Default config instance
CONFIG = Config()

# ---- module-level comment: primary convention ----
# This module uses the Propagating-exception convention:
# Functions raise domain-specific exceptions that propagate to callers.
