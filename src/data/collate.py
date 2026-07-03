"""
# CONVENTION: primary — Propagating-exception convention.

Collate function and DataLoader utilities for variable-length multimodal batches.
"""
import torch
from torch.nn.utils.rnn import pad_sequence


def collate_fn(batch):
    """Collate a list of multimodal samples into a batch.

    Pads time-series modalities to the max length in the batch.
    Stacks frame tensors and labels.

    Args:
        batch: list of sample dicts from HARDataset.__getitem__.

    Returns:
        Dict with batched tensors.
    """
    # Stack labels
    labels = torch.stack([s["label"] for s in batch])
    category_labels = torch.stack([s["category_label"] for s in batch])

    # Stack modality presence flags
    flags = {
        "has_imu": torch.stack([s["has_imu"] for s in batch]),
        "has_radar": torch.stack([s["has_radar"] for s in batch]),
        "has_skeleton": torch.stack([s["has_skeleton"] for s in batch]),
        "has_depth": torch.stack([s["has_depth"] for s in batch]),
        "has_ir": torch.stack([s["has_ir"] for s in batch]),
        "has_thermal": torch.stack([s["has_thermal"] for s in batch]),
    }

    # Pad time-series modalities (IMU, Skeleton) — radar is already uniform shape
    imu_list = [s["imu"] for s in batch if s["imu"].numel() > 0]
    skeleton_list = [s["skeleton"] for s in batch if s["skeleton"].numel() > 0]

    imu_padded, imu_lengths = _pad_1d_sequences(imu_list)
    skeleton_padded, skeleton_lengths = _pad_1d_sequences(skeleton_list)

    # Radar: already padded to fixed shape (F, P, D) by _load_radar — just stack
    radar_list = [s["radar"] for s in batch if s["radar"].numel() > 0]
    radar_stacked = torch.stack(radar_list) if radar_list else torch.empty(0)
    # Radar frame lengths: all frames are valid (no within-batch padding)
    radar_lengths = torch.full((len(radar_list),), radar_list[0].shape[0],
                               dtype=torch.long) if radar_list else torch.empty(0, dtype=torch.long)

    # Stack frame modalities (all have same shape after preprocessing)
    depth_list = [s["depth_color"] for s in batch if s["depth_color"].numel() > 0]
    ir_list = [s["ir"] for s in batch if s["ir"].numel() > 0]
    thermal_list = [s["thermal"] for s in batch if s["thermal"].numel() > 0]

    depth_stacked = torch.stack(depth_list) if depth_list else torch.empty(0)
    ir_stacked = torch.stack(ir_list) if ir_list else torch.empty(0)
    thermal_stacked = torch.stack(thermal_list) if thermal_list else torch.empty(0)

    return {
        "label": labels,
        "category_label": category_labels,
        "imu": imu_padded,
        "imu_lengths": imu_lengths,
        "radar": radar_stacked,
        "radar_lengths": radar_lengths,
        "skeleton": skeleton_padded,
        "skeleton_lengths": skeleton_lengths,
        "depth_color": depth_stacked,
        "ir": ir_stacked,
        "thermal": thermal_stacked,
        "flags": flags,
    }


def _pad_1d_sequences(seq_list):
    """Pad a list of 2D tensors (seq_len, feat_dim) to equal length.

    If feature dimensions vary, pads each sample to the max feat_dim
    so the batch doesn't crash. This is a defensive fallback — the
    data loaders should ensure consistent dims.
    """
    if not seq_list:
        return torch.empty(0), torch.empty(0, dtype=torch.long)

    feat_dims = [s.shape[-1] for s in seq_list]
    max_feat_dim = max(feat_dims)
    min_feat_dim = min(feat_dims)

    # Defensive: align feature dimensions if they vary
    if min_feat_dim != max_feat_dim:
        aligned = []
        for s in seq_list:
            if s.shape[-1] < max_feat_dim:
                pad = torch.zeros(s.shape[0], max_feat_dim - s.shape[-1],
                                  dtype=s.dtype, device=s.device)
                s = torch.cat([s, pad], dim=-1)
            aligned.append(s)
        seq_list = aligned

    feat_dim = max_feat_dim
    lengths = torch.tensor([s.shape[0] for s in seq_list], dtype=torch.long)
    max_len = lengths.max().item()
    batch_size = len(seq_list)

    padded = torch.zeros(batch_size, max_len, feat_dim, dtype=torch.float32)
    for i, seq in enumerate(seq_list):
        l = seq.shape[0]
        padded[i, :l] = seq

    return padded, lengths
