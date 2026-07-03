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
