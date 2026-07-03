"""
# CONVENTION: primary — Propagating-exception convention.

Data module __init__.
"""
from .dataset import HARDataset, discover_clips, build_clip_list
from .collate import collate_fn
from .augmentations import (
    mixup_samples, time_shift, time_stretch, add_gaussian_noise,
    random_frame_dropout, random_temporal_crop
)

__all__ = [
    "HARDataset", "discover_clips", "build_clip_list",
    "collate_fn", "mixup_samples", "time_shift", "time_stretch",
    "add_gaussian_noise", "random_frame_dropout", "random_temporal_crop",
]
