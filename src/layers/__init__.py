"""
# CONVENTION: primary — Propagating-exception convention.
# Guard-clause failures raise exceptions that propagate to callers.

Layers used across modality encoders:
- SEBlock (Squeeze-and-Excitation)
- ResidualSECNNBlock (residual 1D CNN with SE)
- AttentionPooling (learned temporal attention)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .masked_batchnorm import MaskedBatchNorm1d


def create_mask(lengths, max_len):
    """Create a float mask from sequence lengths.

    Args:
        lengths: (batch,) tensor of true sequence lengths.
        max_len: int, maximum sequence length.

    Returns:
        mask: (batch, max_len) float tensor, 1.0 = real, 0.0 = padding.
    """
    batch_size = lengths.size(0)
    mask = torch.arange(max_len, device=lengths.device).expand(
        batch_size, max_len
    ) < lengths.unsqueeze(1)
    return mask.float()


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block with masked average pooling.

    Args:
        channels: number of input channels.
        reduction: reduction ratio for bottleneck.
    """

    MAX_CHANNELS = 4096  # upper-bound for growth inside loop context

    def __init__(self, channels, reduction=8):
        super().__init__()
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x, mask):
        """Apply channel-wise recalibration.

        Args:
            x: (batch, channels, seq_len).
            mask: (batch, seq_len) float mask.

        Returns:
            Recalibrated tensor, same shape as x.
        """
        mask = mask.unsqueeze(1)  # (batch, 1, seq_len)
        masked_x = x * mask
        seq_lengths = mask.sum(dim=-1, keepdim=True)  # (batch, 1, 1)
        y = masked_x.sum(dim=-1, keepdim=True) / (seq_lengths + 1e-8)
        y = self.excitation(y.squeeze(-1)).unsqueeze(-1)
        return x * y.expand_as(x)


class ResidualSECNNBlock(nn.Module):
    """Residual 1D CNN block with Squeeze-and-Excitation.

    Architecture: Conv1d → MaskedBN → ReLU → Conv1d → MaskedBN → SE → +shortcut.

    Args:
        in_channels: input channel count.
        out_channels: output channel count.
        kernel_size: conv kernel size.
        dropout: dropout rate after the block.
        weight_decay: stored for reference (not used in forward).
    """

    MAX_OUT_CHANNELS = 4096  # upper-bound guard

    def __init__(self, in_channels, out_channels, kernel_size, dropout=0.3, weight_decay=1e-4):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=kernel_size // 2, bias=False
        )
        self.bn1 = MaskedBatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=kernel_size // 2, bias=False
        )
        self.bn2 = MaskedBatchNorm1d(out_channels)
        self.se = SEBlock(out_channels)
        self.shortcut = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        self.shortcut_bn = MaskedBatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        """Forward pass with residual connection.

        Args:
            x: (batch, in_channels, seq_len).
            mask: (batch, seq_len) float mask.

        Returns:
            (batch, out_channels, seq_len).
        """
        # Shortcut
        shortcut = self.shortcut(x) * mask.unsqueeze(1)
        shortcut = self.shortcut_bn(shortcut, mask)

        # Double conv
        out = self.conv1(x) * mask.unsqueeze(1)
        out = F.relu(self.bn1(out, mask))
        out = self.conv2(out) * mask.unsqueeze(1)
        out = self.bn2(out, mask)

        # SE recalibration
        out = self.se(out, mask)

        # Residual + activation + dropout
        out = out + shortcut
        out = F.relu(out)
        out = self.dropout(out) * mask.unsqueeze(1)

        return out


class AttentionPooling(nn.Module):
    """Learned attention-weighted temporal pooling.

    Args:
        hidden_dim: feature dimension.
    """

    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, x, mask):
        """Pool sequence over time dimension.

        Args:
            x: (batch, seq_len, hidden_dim).
            mask: (batch, seq_len) float mask.

        Returns:
            (batch, hidden_dim) pooled feature vector.
        """
        scores = self.attention(x).squeeze(-1)  # (batch, seq_len)
        # Use the dtype's own min (not a hardcoded -1e9) so this stays finite
        # under AMP, where scores is float16 and -1e9 overflows to -inf/NaN.
        scores = scores.masked_fill(~mask.bool(), torch.finfo(scores.dtype).min)
        weights = F.softmax(scores, dim=1)
        weights = weights * mask
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
        context = torch.sum(x * weights.unsqueeze(-1), dim=1)
        return context


class PhaseAttentionLayer(nn.Module):
    """Phase-aware attention pooling (from CMI 2nd place).

    Combines learned attention with externally-provided phase weights.

    Args:
        hidden_dim: feature dimension.
    """

    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, x, phase_weights, mask):
        """Pool using combined attention + phase weights.

        Args:
            x: (batch, seq_len, hidden_dim).
            phase_weights: (batch, seq_len) per-timestep phase scores.
            mask: (batch, seq_len) float mask.

        Returns:
            (batch, hidden_dim).
        """
        scores = torch.tanh(self.attention(x)).squeeze(-1)
        scores = scores.masked_fill(~mask.bool(), torch.finfo(scores.dtype).min)
        attention_weights = F.softmax(scores, dim=1)
        combined = attention_weights * phase_weights * mask
        combined = combined / (combined.sum(dim=1, keepdim=True) + 1e-8)
        context = torch.sum(x * combined.unsqueeze(-1), dim=1)
        return context


class SegmentPooling(nn.Module):
    """Multi-segment temporal pooling (CMI 6th place pattern).

    Splits the time dimension into N equal segments, pools each one via
    attention, and concatenates. Captures early/mid/late temporal dynamics
    without requiring real phase labels.

    Args:
        hidden_dim: feature dimension.
        n_segments: number of time segments (default 3).
    """

    MAX_SEGMENTS = 10  # upper-bound guard

    def __init__(self, hidden_dim, n_segments=3):
        super().__init__()
        self.n_segments = n_segments
        self.attentions = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(n_segments)
        ])

    def forward(self, x, mask):
        """Pool each time segment independently, then concatenate.

        Args:
            x: (batch, seq_len, hidden_dim).
            mask: (batch, seq_len) float mask.

        Returns:
            (batch, n_segments * hidden_dim) pooled features.
        """
        B, T, D = x.shape
        segment_outputs = []
        seg_len = max(1, T // self.n_segments)

        for seg_idx in range(self.n_segments):
            t_start = seg_idx * seg_len
            t_end = min(T, (seg_idx + 1) * seg_len) if seg_idx < self.n_segments - 1 else T

            if t_start >= t_end:
                continue

            x_seg = x[:, t_start:t_end, :]          # (B, seg_len, D)
            m_seg = mask[:, t_start:t_end]           # (B, seg_len)

            scores = self.attentions[seg_idx](x_seg).squeeze(-1)
            scores = scores.masked_fill(~m_seg.bool(), torch.finfo(scores.dtype).min)
            weights = F.softmax(scores, dim=1)
            weights = weights * m_seg
            weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
            seg_out = torch.sum(x_seg * weights.unsqueeze(-1), dim=1)  # (B, D)
            segment_outputs.append(seg_out)

        return torch.cat(segment_outputs, dim=-1)  # (B, n_seg * D)


class CrossModalFusion(nn.Module):
    """Lets modality-level embeddings attend to each other before fusion,
    instead of pure late-fusion concatenation (CMI 1st place pattern).

    Operates over a short "sequence" of one already-pooled vector per
    modality (length = num_modalities, typically 6) — computationally
    trivial compared to per-timestep cross-attention across each
    modality's full (and very differently-lengthed: IMU 128, Skeleton 42,
    Radar 82, frames 16-32) time axis, but still lets e.g. the IMU vector
    be modulated by what the Skeleton vector contains (and vice versa)
    before the classifier ever sees them — something plain concatenation
    + MLP can only approximate indirectly through the MLP's weights, never
    directly conditioning one modality's representation on another's.

    Standard single-layer Transformer-encoder block: multi-head
    self-attention + residual + layernorm, then a small feedforward +
    residual + layernorm. Missing modalities don't need special handling
    here — the caller substitutes them with a learned "missing" token
    before this module ever sees the sequence (see FusionHead), so every
    position is already a well-formed vector.

    Args:
        embed_dim: dimension of each modality's embedding. Must be
            divisible by num_heads.
        num_heads: attention heads.
        dropout: dropout rate (applied inside attention and the feedforward).
    """

    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, modality_seq):
        """
        Args:
            modality_seq: (batch, num_modalities, embed_dim).

        Returns:
            (batch, num_modalities, embed_dim) — same shape, each
            modality's vector now informed by cross-attention over all
            modalities (including itself, standard self-attention).
        """
        attn_out, _ = self.attn(modality_seq, modality_seq, modality_seq)
        x = self.norm1(modality_seq + attn_out)
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x


class TimeAlignedFrameCrossAttention(nn.Module):
    """Cross-attention between Depth_Color, IR, and Thermal at matching
    real-world time positions, applied to each modality's per-frame
    sequence BEFORE its own temporal pooling collapses time away — this is
    the actual CMI 1st place pattern (see public_solution_ogurtsov's
    IMUCrossAttentionFusion, which cross-attends (B, T, C) sequences with T
    intact), as opposed to CrossModalFusion in this same file, which only
    attends over already-pooled single vectors and is a shallower
    approximation of it.

    This exact time alignment is only possible for these 3 modalities:
    Depth_Color and IR both uniformly sample config.num_frames frames
    across the clip; Thermal uniformly samples config.num_thermal_frames =
    2 * num_frames frames across the *same* clip (native fps is ~2.5x, but
    the sampled counts land on an exact 2:1 ratio) — so Depth_Color[i] and
    IR[i] represent the same instant Thermal[2i] and Thermal[2i+1]
    straddle. IMU/Radar/Skeleton don't share a common divisor fine enough
    to align this way (deliberately out of scope here — see the ablation
    flag name).

    Each time-bin's group of exactly 4 embeddings (1 depth + 1 ir + 2
    thermal) attends only within itself, not across bins — vectorized as
    one batched attention call over an effective batch of batch*num_bins.

    Missing modalities are masked out of the attention's key/value side
    (nn.MultiheadAttention's key_padding_mask) so a sample missing e.g.
    Depth_Color doesn't let IR/Thermal attend to its zero-filled frames —
    unlike CrossModalFusion's post-pooling case (where missing-modality
    substitution already happened before it ever runs), this module
    operates on raw per-frame CNN output, which has no such substitution
    yet, so it needs its own masking.
    """

    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, depth_seq, ir_seq, thermal_seq, has_depth, has_ir, has_thermal):
        """
        Args:
            depth_seq: (B, N, D) per-frame Depth_Color embeddings (pre-pool).
            ir_seq: (B, N, D) per-frame IR embeddings (pre-pool).
            thermal_seq: (B, 2N, D) per-frame Thermal embeddings (pre-pool).
            has_depth, has_ir, has_thermal: (B,) float or bool presence flags.

        Returns:
            (depth_enhanced, ir_enhanced, thermal_enhanced), same shapes
            as the corresponding inputs.
        """
        B, N, D = depth_seq.shape

        # Group into (B, N, 4, D): depth, ir, thermal[2i], thermal[2i+1].
        thermal_pairs = thermal_seq.view(B, N, 2, D)
        group = torch.cat([
            depth_seq.unsqueeze(2),   # (B, N, 1, D)
            ir_seq.unsqueeze(2),      # (B, N, 1, D)
            thermal_pairs,            # (B, N, 2, D)
        ], dim=2)                     # (B, N, 4, D)

        # Flatten batch*time-bins into one effective batch dimension so a
        # single batched attention call handles all N bins at once, rather
        # than looping over bins in Python.
        group_flat = group.reshape(B * N, 4, D)

        # key_padding_mask convention: True = ignore this position. The
        # same per-sample presence applies to every one of its N bins.
        absent = torch.stack([
            ~has_depth.bool(), ~has_ir.bool(), ~has_thermal.bool(), ~has_thermal.bool(),
        ], dim=-1)                                          # (B, 4)
        key_padding_mask = absent.unsqueeze(1).expand(B, N, 4).reshape(B * N, 4)

        # Guard against an all-masked row (all 3 modalities missing at
        # once for that sample): softmax over an entirely -inf row is
        # undefined. Rare/unlikely given the dataset's ~95.9% all-present
        # rate, but cheap insurance against NaN propagation regardless —
        # unmask everything for those rows instead (their outputs get
        # discarded downstream by the per-modality missing-token
        # substitution anyway, same as the fully-empty-batch case
        # elsewhere in this codebase).
        fully_absent = key_padding_mask.all(dim=-1, keepdim=True)
        key_padding_mask = key_padding_mask & ~fully_absent

        attn_out, _ = self.attn(
            group_flat, group_flat, group_flat, key_padding_mask=key_padding_mask
        )
        x = self.norm1(group_flat + attn_out)
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)   # (B*N, 4, D)

        x = x.view(B, N, 4, D)
        depth_enhanced = x[:, :, 0, :]
        ir_enhanced = x[:, :, 1, :]
        thermal_enhanced = x[:, :, 2:4, :].reshape(B, N * 2, D)

        return depth_enhanced, ir_enhanced, thermal_enhanced
