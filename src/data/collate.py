"""
# CONVENTION: primary — Propagating-exception convention.

Collate function and DataLoader utilities for multimodal batches.
"""
import torch


def collate_fn(batch):
    """Collate a list of multimodal samples into a batch.

    Every per-sample modality tensor produced by HARDataset has a fixed,
    config-determined shape — whether the modality is genuinely present or
    is a zero placeholder for a missing/corrupted clip — so every modality
    can be stacked directly. This keeps row `i` of every tensor (including
    `flags` and `label`) aligned with sample `i` of the batch; previously
    missing-modality samples were filtered out of that modality's tensor
    only, which desynced it from the rest of the batch.

    Args:
        batch: list of sample dicts from HARDataset.__getitem__.

    Returns:
        Dict with batched tensors.
    """
    labels = torch.stack([s["label"] for s in batch])
    category_labels = torch.stack([s["category_label"] for s in batch])

    flags = {
        "has_imu": torch.stack([s["has_imu"] for s in batch]),
        "has_radar": torch.stack([s["has_radar"] for s in batch]),
        "has_skeleton": torch.stack([s["has_skeleton"] for s in batch]),
        "has_depth": torch.stack([s["has_depth"] for s in batch]),
        "has_ir": torch.stack([s["has_ir"] for s in batch]),
        "has_thermal": torch.stack([s["has_thermal"] for s in batch]),
    }

    imu_stacked = torch.stack([s["imu"] for s in batch])
    imu_lengths = torch.full((len(batch),), imu_stacked.shape[1], dtype=torch.long)

    skeleton_stacked = torch.stack([s["skeleton"] for s in batch])
    skeleton_lengths = torch.full((len(batch),), skeleton_stacked.shape[1], dtype=torch.long)

    radar_stacked = torch.stack([s["radar"] for s in batch])
    radar_lengths = torch.full((len(batch),), radar_stacked.shape[1], dtype=torch.long)

    depth_stacked = torch.stack([s["depth_color"] for s in batch])
    ir_stacked = torch.stack([s["ir"] for s in batch])
    thermal_stacked = torch.stack([s["thermal"] for s in batch])

    return {
        "label": labels,
        "category_label": category_labels,
        "imu": imu_stacked,
        "imu_lengths": imu_lengths,
        "radar": radar_stacked,
        "radar_lengths": radar_lengths,
        "skeleton": skeleton_stacked,
        "skeleton_lengths": skeleton_lengths,
        "depth_color": depth_stacked,
        "ir": ir_stacked,
        "thermal": thermal_stacked,
        "flags": flags,
    }
