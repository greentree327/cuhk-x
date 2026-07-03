"""
# CONVENTION: primary — Propagating-exception convention.

Augmentation functions for time-series and multimodal data.
"""
import random
import numpy as np
import torch


# Upper-bound constants for loop guards
MAX_AUG_ATTEMPTS = 10
MAX_SEQ_LEN = 4096


def mixup_samples(x1, x2, y1, y2, alpha=0.3, num_classes=40):
    """Apply Mixup augmentation to two samples.

    Args:
        x1, x2: feature tensors of shape (seq_len, feat_dim) or (C, H, W).
        y1, y2: integer class labels.
        alpha: Beta distribution parameter.
        num_classes: number of classes.

    Returns:
        mixed_x, mixed_y (soft label as one-hot).
    """
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    mixed_x = lam * x1 + (1 - lam) * x2
    y1_oh = torch.zeros(num_classes)
    y2_oh = torch.zeros(num_classes)
    y1_oh[y1] = 1.0
    y2_oh[y2] = 1.0
    mixed_y = lam * y1_oh + (1 - lam) * y2_oh
    return mixed_x, mixed_y


def time_shift(x, max_shift_pct=0.25):
    """Randomly shift a 1D time-series along the time axis.

    Args:
        x: (seq_len, feat_dim) tensor.
        max_shift_pct: maximum shift as fraction of sequence length.

    Returns:
        Shifted tensor, same shape.
    """
    seq_len = x.shape[0]
    max_shift = int(seq_len * max_shift_pct)
    if max_shift < 1:
        return x
    shift = random.randint(-max_shift, max_shift)
    if shift > 0:
        # Shift right: pad left with zeros
        x = torch.cat([torch.zeros(shift, x.shape[1]), x[:-shift]], dim=0)
    elif shift < 0:
        # Shift left: pad right with zeros
        shift = -shift
        x = torch.cat([x[shift:], torch.zeros(shift, x.shape[1])], dim=0)
    return x


def time_stretch(x, stretch_range=(0.8, 1.2)):
    """Randomly stretch or compress a time-series via interpolation.

    Args:
        x: (seq_len, feat_dim) tensor.
        stretch_range: (min, max) stretch factor.

    Returns:
        Stretched tensor, same shape as input.
    """
    seq_len = x.shape[0]
    factor = random.uniform(*stretch_range)
    new_len = max(1, int(seq_len * factor))
    indices = torch.linspace(0, seq_len - 1, new_len)
    indices_floor = indices.long()
    indices_ceil = torch.clamp(indices_floor + 1, max=seq_len - 1)
    alpha = (indices - indices_floor.float()).unsqueeze(-1)

    stretched = x[indices_floor] * (1 - alpha) + x[indices_ceil] * alpha

    # Resize back to original length
    if new_len > seq_len:
        stretched = stretched[:seq_len]
    elif new_len < seq_len:
        pad = torch.zeros(seq_len - new_len, x.shape[1])
        stretched = torch.cat([stretched, pad], dim=0)

    return stretched


def add_gaussian_noise(x, std=0.01):
    """Add Gaussian noise to a tensor.

    Args:
        x: tensor of any shape.
        std: standard deviation of noise.

    Returns:
        Noisy tensor, same shape.
    """
    noise = torch.randn_like(x) * std
    return x + noise


def random_frame_dropout(frames, drop_prob=0.1):
    """Randomly zero out individual frames.

    Args:
        frames: (N_frames, C, H, W) tensor.
        drop_prob: probability of dropping each frame.

    Returns:
        Tensor with some frames zeroed, same shape.
    """
    mask = (torch.rand(frames.shape[0]) > drop_prob).float()
    mask = mask.view(-1, 1, 1, 1)
    return frames * mask


def random_temporal_crop(x, output_len):
    """Randomly crop a contiguous segment from a time-series.

    Args:
        x: (seq_len, feat_dim) tensor.
        output_len: desired output length.

    Returns:
        (output_len, feat_dim) tensor.
        If seq_len < output_len, pads with zeros at the end.
    """
    seq_len = x.shape[0]
    if seq_len <= output_len:
        pad = torch.zeros(output_len - seq_len, x.shape[1])
        return torch.cat([x, pad], dim=0)
    start = random.randint(0, seq_len - output_len)
    return x[start : start + output_len]
