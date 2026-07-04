"""
# CONVENTION: primary — Propagating-exception convention.

Modality-specific encoders for the CUHK-X Small Model Track.

Encoders:
- IMUEncoder: Residual SE-CNN with multi-branch per feature group.
- FrameEncoder: 2D CNN backbone + temporal attention pooling.
- RadarEncoder: PointNet-lite per frame + 1D CNN temporal.
- SkeletonEncoder: 1D CNN over flattened joint coordinates.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path

# Allow importing from parent
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from layers import ResidualSECNNBlock, AttentionPooling, create_mask, SegmentPooling


class IMUEncoder(nn.Module):
    """IMU encoder with grouped feature branches (inspired by CMI 2nd place).

    Processes IMU features in 3 sub-branches (acc, gyro/rotation, derived),
    then fuses via a final SE-CNN block and attention pooling.

    Args:
        input_dim: total IMU feature dimension (all 5 sensors concatenated).
        encoder_dim: output embedding dimension.
        dropout: dropout rate.
        weight_decay: stored for reference.
    """

    MAX_BRANCHES = 6  # upper bound for branch count

    def __init__(self, input_dim=135, encoder_dim=256, dropout=0.3, weight_decay=1e-4,
                 use_segment_pooling=False, n_segments=3):
        super().__init__()
        self.input_dim = input_dim
        self.encoder_dim = encoder_dim

        # Channel split: 1/3 acc-deriv, 1/3 rot-gyro, 1/3 rest
        acc_dim = input_dim // 3
        gyro_dim = input_dim // 3
        derived_dim = input_dim - acc_dim - gyro_dim

        self.acc_dim = acc_dim
        self.gyro_dim = gyro_dim
        self.derived_dim = derived_dim

        # Sub-branches
        self.acc_branch = nn.Sequential(
            ResidualSECNNBlock(acc_dim, 64, 1, dropout, weight_decay),
            ResidualSECNNBlock(64, 128, 3, dropout, weight_decay),
            ResidualSECNNBlock(128, 256, 5, dropout, weight_decay),
        )
        self.gyro_branch = nn.Sequential(
            ResidualSECNNBlock(gyro_dim, 64, 1, dropout, weight_decay),
            ResidualSECNNBlock(64, 128, 3, dropout, weight_decay),
            ResidualSECNNBlock(128, 256, 5, dropout, weight_decay),
        )
        self.derived_branch = nn.Sequential(
            ResidualSECNNBlock(derived_dim, 64, 1, dropout, weight_decay),
            ResidualSECNNBlock(64, 128, 3, dropout, weight_decay),
            ResidualSECNNBlock(128, 256, 5, dropout, weight_decay),
        )

        # Fusion block
        self.fusion = ResidualSECNNBlock(256 * 3, encoder_dim, 3, dropout, weight_decay)

        # Attention pooling (segment pooling splits the pooled output into
        # n_segments early/mid/late chunks instead of one vector — see
        # SegmentPooling's docstring. pool_out_dim is what downstream fusion
        # code needs to know to size itself correctly.)
        self.use_segment_pooling = use_segment_pooling
        if use_segment_pooling:
            self.pool = SegmentPooling(encoder_dim, n_segments)
            self.pool_out_dim = encoder_dim * n_segments
        else:
            self.pool = AttentionPooling(encoder_dim)
            self.pool_out_dim = encoder_dim

    def forward(self, x, lengths):
        """Forward pass.

        Args:
            x: (batch, seq_len, input_dim) float tensor.
            lengths: (batch,) long tensor of true sequence lengths.

        Returns:
            (batch, pool_out_dim) embedding.
        """
        x = x.transpose(1, 2)  # (batch, input_dim, seq_len)
        seq_len = x.size(2)
        mask = create_mask(lengths, seq_len)

        # Split channels
        x_acc = x[:, :self.acc_dim, :]
        x_gyro = x[:, self.acc_dim:self.acc_dim + self.gyro_dim, :]
        x_derived = x[:, self.acc_dim + self.gyro_dim:, :]

        # Process sub-branches
        for layer in self.acc_branch:
            x_acc = layer(x_acc, mask)
        for layer in self.gyro_branch:
            x_gyro = layer(x_gyro, mask)
        for layer in self.derived_branch:
            x_derived = layer(x_derived, mask)

        # Fuse
        x = torch.cat([x_acc, x_gyro, x_derived], dim=1)
        x = self.fusion(x, mask)

        # Pool
        x = x.transpose(1, 2)  # (batch, seq_len, encoder_dim)
        x = self.pool(x, mask)

        return x


class FrameEncoder(nn.Module):
    """Lightweight frame encoder trained from scratch.

    Competition rule: NO pretrained backbones allowed.
    Uses a custom 4-stage conv net with double-conv per stage.
    ~0.5M params per modality instance (3 for Depth/IR/Thermal ≈ 1.5M total).

    Architecture:
        Stage1: (C, H, W) → (base, H/2, W/2)
        Stage2: → (base*2, H/4, W/4)
        Stage3: → (base*4, H/8, W/8)
        Stage4: → (base*8, H/16, W/16)
        Head: AdaptiveAvgPool2d(1) → Flatten → Linear → BN → ReLU → Dropout
        Temporal: AttentionPooling over N sampled frames.

    Args:
        in_channels: 3 for Depth_Color/Thermal, 1 for IR (grayscale).
        encoder_dim: output embedding dimension (default 256).
        base_width: base channel count, doubles each stage. Default 32.
        dropout: dropout rate.
        use_segment_pooling: split pooling into early/mid/late chunks (see
            SegmentPooling). Frame modalities always sample a fixed frame
            count uniformly from however many native frames exist, so
            (unlike IMU/Skeleton/Radar, whose fixed-length sequences can be
            mostly trailing zero-padding for short clips) segments here are
            usually real content end-to-end rather than padding.
        n_segments: number of segments when use_segment_pooling=True.
    """

    MAX_FRAMES = 256  # upper bound for frame count

    def __init__(self, in_channels=3, encoder_dim=256, base_width=32, dropout=0.3,
                 use_segment_pooling=False, n_segments=3):
        super().__init__()
        self.encoder_dim = encoder_dim

        # 4-stage conv net (all trained from scratch)
        self.stage1 = self._conv_block(in_channels, base_width, stride=2)
        self.stage2 = self._conv_block(base_width, base_width * 2, stride=2)
        self.stage3 = self._conv_block(base_width * 2, base_width * 4, stride=2)
        self.stage4 = self._conv_block(base_width * 4, base_width * 8, stride=2)

        bb_dim = base_width * 8  # 256 when base_width=32

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(bb_dim, encoder_dim, bias=False),
            nn.BatchNorm1d(encoder_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # Temporal pooling
        self.use_segment_pooling = use_segment_pooling
        if use_segment_pooling:
            self.pool = SegmentPooling(encoder_dim, n_segments)
            self.pool_out_dim = encoder_dim * n_segments
        else:
            self.pool = AttentionPooling(encoder_dim)
            self.pool_out_dim = encoder_dim

    @staticmethod
    def _conv_block(in_ch, out_ch, stride=2):
        """Double-conv block: Conv3x3 → BN → ReLU → Conv3x3 → BN → ReLU."""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, frames):
        """Forward pass.

        Args:
            frames: (batch, N_frames, C, H, W) float tensor.
                   C=3 for Depth_Color/Thermal, C=1 for IR.

        Returns:
            (batch, pool_out_dim) embedding.
            Returns zero tensor of correct shape if frames is empty.
        """
        if frames.numel() == 0:
            return torch.zeros(0, self.pool_out_dim, device=frames.device)

        B, N, C, H, W = frames.shape
        # Flatten batch+frame dims for shared conv processing
        frames_flat = frames.view(B * N, C, H, W)
        x = self.stage1(frames_flat)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.head(x)        # (B*N, encoder_dim)
        x = x.view(B, N, -1)    # (B, N, encoder_dim)

        # Temporal attention pooling
        mask = torch.ones(B, N, device=frames.device)
        pooled = self.pool(x, mask)

        return pooled


class RadarEncoder(nn.Module):
    """Radar point-cloud encoder: PointNet-lite per frame + 1D CNN temporal.

    Args:
        point_dim: features per radar point (x, y, z, doppler, intensity).
        max_points: max points per frame.
        encoder_dim: output embedding dimension.
        dropout: dropout rate.
        weight_decay: L2 regularization.
        use_segment_pooling: split pooling into early/mid/late chunks (see
            SegmentPooling).
        n_segments: number of segments when use_segment_pooling=True.
    """

    MAX_POINTS = 512  # upper bound

    def __init__(self, point_dim=6, max_points=64, encoder_dim=256,
                 dropout=0.3, weight_decay=1e-4,
                 use_segment_pooling=False, n_segments=3):
        super().__init__()
        self.max_points = max_points
        self.encoder_dim = encoder_dim

        # PointNet MLP (shared per-point)
        self.point_mlp = nn.Sequential(
            nn.Linear(point_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
        )

        # Temporal 1D CNN after per-frame features
        self.temporal = nn.Sequential(
            ResidualSECNNBlock(256, 128, 3, dropout, weight_decay),
            ResidualSECNNBlock(128, encoder_dim, 5, dropout, weight_decay),
        )

        self.use_segment_pooling = use_segment_pooling
        if use_segment_pooling:
            self.pool = SegmentPooling(encoder_dim, n_segments)
            self.pool_out_dim = encoder_dim * n_segments
        else:
            self.pool = AttentionPooling(encoder_dim)
            self.pool_out_dim = encoder_dim

    def forward(self, radar_data, lengths):
        """Forward pass.

        Args:
            radar_data: (batch, n_frames, max_points, point_dim) float tensor.
            lengths: (batch,) long tensor of true frame counts.

        Returns:
            (batch, pool_out_dim) embedding.
        """
        if radar_data.numel() == 0:
            return torch.zeros(0, self.pool_out_dim, device=radar_data.device)

        B, F, P, D = radar_data.shape

        # PointNet per frame: (B*F, P, D) → (B*F, 256)
        points_flat = radar_data.view(B * F, P, D)
        per_point = self.point_mlp(points_flat)     # (B*F, P, 256)
        per_frame = per_point.max(dim=1).values      # (B*F, 256)
        per_frame = per_frame.view(B, F, 256)        # (B, F, 256)

        # Temporal CNN
        per_frame = per_frame.transpose(1, 2)        # (B, 256, F)
        mask = create_mask(lengths, F)
        for layer in self.temporal:
            per_frame = layer(per_frame, mask)

        # Pool
        per_frame = per_frame.transpose(1, 2)        # (B, F, encoder_dim)
        pooled = self.pool(per_frame, mask)

        return pooled


class SkeletonEncoder(nn.Module):
    """Skeleton encoder: flatten joints → 1D CNN → attention pooling.

    For 17 joints × 3D coords = 51 input features per frame.

    Args:
        num_joints: number of skeleton joints.
        joint_dim: coordinates per joint (3 for x,y,z).
        encoder_dim: output embedding dimension.
        dropout: dropout rate.
        weight_decay: L2 regularization.
    """

    MAX_JOINTS = 64  # upper bound

    def __init__(self, num_joints=17, joint_dim=3, encoder_dim=256,
                 dropout=0.3, weight_decay=1e-4,
                 use_segment_pooling=False, n_segments=3, input_dim=None):
        super().__init__()
        if input_dim is None:
            input_dim = num_joints * joint_dim  # 51 default
        self.input_dim = input_dim
        self.encoder_dim = encoder_dim

        self.cnn = nn.Sequential(
            ResidualSECNNBlock(input_dim, 128, 3, dropout, weight_decay),
            ResidualSECNNBlock(128, 256, 5, dropout, weight_decay),
            ResidualSECNNBlock(256, encoder_dim, 7, dropout, weight_decay),
        )

        self.use_segment_pooling = use_segment_pooling
        if use_segment_pooling:
            self.pool = SegmentPooling(encoder_dim, n_segments)
            self.pool_out_dim = encoder_dim * n_segments
        else:
            self.pool = AttentionPooling(encoder_dim)
            self.pool_out_dim = encoder_dim

    def forward(self, skeleton, lengths):
        """Forward pass.

        Args:
            skeleton: (batch, seq_len, num_joints * joint_dim) float tensor.
            lengths: (batch,) long tensor of true sequence lengths.

        Returns:
            (batch, pool_out_dim) embedding.
        """
        if skeleton.numel() == 0:
            return torch.zeros(0, self.pool_out_dim, device=skeleton.device)

        x = skeleton.transpose(1, 2)  # (batch, input_dim, seq_len)
        seq_len = x.size(2)
        mask = create_mask(lengths, seq_len)

        for layer in self.cnn:
            x = layer(x, mask)

        x = x.transpose(1, 2)  # (batch, seq_len, encoder_dim)
        pooled = self.pool(x, mask)

        return pooled


# ---- Module-level bounds verification ----
_FEATURE_DIMS = {
    "IMUEncoder": 60,
    "FrameEncoder": 256,
    "RadarEncoder": 256,
    "SkeletonEncoder": 256,
}
