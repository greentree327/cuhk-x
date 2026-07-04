"""
# CONVENTION: primary — Propagating-exception convention.

Fusion head and full multimodal HAR model.

FusionHead: concatenates modality embeddings → MLP → 40-class logits.
HARModel: full model combining all 6 encoders + fusion head + auxiliary head.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from .encoders import IMUEncoder, FrameEncoder, RadarEncoder, SkeletonEncoder
from layers import CrossModalFusion


class FusionHead(nn.Module):
    """Late-fusion MLP: concatenate modality embeddings → classify.

    Handles missing modalities via learned "missing" embedding tokens.

    Args:
        num_modalities: number of modalities (6).
        encoder_dim: dimension of each modality embedding.
        hidden_dim: hidden dimension in fusion MLP.
        num_classes: number of action classes.
        dropout: dropout rate.
        use_cross_modal_attention: let modality embeddings attend to each
            other (CrossModalFusion) before concatenation, instead of pure
            late fusion. See CrossModalFusion's docstring.
    """

    MAX_MODALITIES = 10  # upper bound

    def __init__(self, num_modalities=6, encoder_dim=256, hidden_dim=512,
                 num_classes=40, dropout=0.3, use_cross_modal_attention=False):
        super().__init__()
        self.num_modalities = num_modalities
        self.encoder_dim = encoder_dim
        self.use_cross_modal_attention = use_cross_modal_attention

        # Learned "missing modality" embedding tokens
        self.missing_tokens = nn.ParameterList([
            nn.Parameter(torch.zeros(1, encoder_dim))
            for _ in range(num_modalities)
        ])
        # Initialize with small random values
        for token in self.missing_tokens:
            nn.init.normal_(token, std=0.02)

        if use_cross_modal_attention:
            self.cross_modal = CrossModalFusion(encoder_dim, num_heads=4, dropout=dropout)

        concat_dim = num_modalities * encoder_dim

        self.mlp = nn.Sequential(
            nn.Linear(concat_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2, bias=False),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.7),
        )

        self.classifier = nn.Linear(hidden_dim // 2, num_classes)

    def forward(self, embeddings, flags, return_penultimate=False):
        """Forward pass with missing modality handling.

        Args:
            embeddings: list of (batch, encoder_dim) tensors, one per modality.
            flags: (batch, num_modalities) float tensor, 1 = present, 0 = missing.
            return_penultimate: if True, also return the 256-dim feature before classifier.

        Returns:
            logits: (batch, num_classes).
            penultimate: (batch, hidden//2) if return_penultimate, else None.
        """
        batch_size = flags.shape[0]
        fused = []

        for i, emb in enumerate(embeddings):
            token = self.missing_tokens[i].expand(batch_size, -1)
            if emb.numel() == 0 or emb.shape[0] == 0:
                # Modality absent for the entire batch (e.g. never present
                # in the dataset at all) — use the learned token outright.
                fused.append(token)
            else:
                # Per-sample substitution: rows where this modality is
                # missing for that sample get the learned token instead of
                # the encoder's output on zero-placeholder input.
                present = flags[:, i].unsqueeze(-1)  # (batch, 1)
                fused.append(present * emb + (1.0 - present) * token)

        if self.use_cross_modal_attention:
            # Stack into a length-num_modalities "sequence" so each
            # modality's vector can attend to every other modality's vector
            # before fusion, then flatten back to the same shape
            # torch.cat(fused) would have produced.
            modality_seq = torch.stack(fused, dim=1)   # (B, num_modalities, encoder_dim)
            modality_seq = self.cross_modal(modality_seq)
            x = modality_seq.reshape(batch_size, -1)   # (B, num_modalities * encoder_dim)
        else:
            x = torch.cat(fused, dim=-1)

        penultimate = self.mlp(x)
        logits = self.classifier(penultimate)

        if return_penultimate:
            return logits, penultimate
        return logits, None


class HARModel(nn.Module):
    """Full multimodal Human Action Recognition model.

    Combines IMU, Radar, Skeleton, Depth_Color, IR, and Thermal encoders
    with a late-fusion head.

    Args:
        config: Config object with model/training settings.
    """

    MAX_EMBEDDINGS = 6  # upper bound for modality count

    def __init__(self, config):
        super().__init__()
        self.config = config
        enc_dim = config.encoder_dim

        # Time-series encoders
        seg_pool = config.flags.use_segment_pooling
        n_seg = config.flags.segment_count
        self.imu_encoder = IMUEncoder(
            input_dim=config.imu_input_dim, encoder_dim=enc_dim,
            dropout=config.dropout, weight_decay=config.weight_decay,
            use_segment_pooling=seg_pool, n_segments=n_seg,
        )
        self.radar_encoder = RadarEncoder(
            point_dim=config.radar_point_dim, max_points=config.radar_max_points,
            encoder_dim=enc_dim, dropout=config.dropout,
            weight_decay=config.weight_decay,
            use_segment_pooling=seg_pool, n_segments=n_seg,
        )
        self.skeleton_encoder = SkeletonEncoder(
            num_joints=config.skel_num_joints, joint_dim=config.skel_joint_dim,
            encoder_dim=enc_dim, dropout=config.dropout,
            weight_decay=config.weight_decay,
            use_segment_pooling=seg_pool, n_segments=n_seg,
            input_dim=config.skel_input_dim,
        )

        # Frame encoders (separate instances per modality, trained from scratch)
        # Depth_Color and Thermal are 3-channel, IR is 1-channel (grayscale)
        self.depth_encoder = FrameEncoder(
            in_channels=3, encoder_dim=enc_dim, dropout=config.dropout,
            use_segment_pooling=seg_pool, n_segments=n_seg,
        )
        self.ir_encoder = FrameEncoder(
            in_channels=1, encoder_dim=enc_dim, dropout=config.dropout,
            use_segment_pooling=seg_pool, n_segments=n_seg,
        )
        self.thermal_encoder = FrameEncoder(
            in_channels=3, encoder_dim=enc_dim, dropout=config.dropout,
            use_segment_pooling=seg_pool, n_segments=n_seg,
        )

        # All 6 encoders are constructed with the same seg_pool/n_seg toggle,
        # so they share one effective per-modality embedding dimension:
        # enc_dim normally, or enc_dim * n_seg when segment pooling is on
        # (SegmentPooling concatenates rather than averages its segments —
        # see layers/__init__.py). FusionHead and CrossModalFusion both need
        # to know this to size their layers correctly.
        effective_dim = self.imu_encoder.pool_out_dim

        # Fusion
        self.fusion = FusionHead(
            num_modalities=6, encoder_dim=effective_dim,
            hidden_dim=config.fusion_hidden, num_classes=config.num_classes,
            dropout=config.dropout,
            use_cross_modal_attention=config.flags.use_cross_modal_attention,
        )

        # Auxiliary head: predict coarse action category
        self.num_categories = config.num_categories
        self.aux_head = nn.Linear(config.fusion_hidden // 2, self.num_categories)

    def forward(self, batch, return_embeddings=False):
        """Full forward pass.

        Args:
            batch: dict from collate_fn with modality tensors.
            return_embeddings: if True, also returns per-modality embeddings.

        Returns:
            dict with keys:
                - logits: (batch, 40) action logits.
                - aux_logits: (batch, 8) category logits (if aux enabled).
                - embeddings: list of (batch, enc_dim) (if return_embeddings).
        """
        embeddings = []
        flags = batch["flags"]

        # IMU
        if batch["imu"].numel() > 0:
            emb = self.imu_encoder(batch["imu"], batch["imu_lengths"])
        else:
            emb = torch.empty(0)
        embeddings.append(emb)

        # Radar
        if batch["radar"].numel() > 0:
            emb = self.radar_encoder(batch["radar"], batch["radar_lengths"])
        else:
            emb = torch.empty(0)
        embeddings.append(emb)

        # Skeleton
        if batch["skeleton"].numel() > 0:
            emb = self.skeleton_encoder(batch["skeleton"], batch["skeleton_lengths"])
        else:
            emb = torch.empty(0)
        embeddings.append(emb)

        # Depth_Color
        if batch["depth_color"].numel() > 0:
            emb = self.depth_encoder(batch["depth_color"])
        else:
            emb = torch.empty(0)
        embeddings.append(emb)

        # IR
        if batch["ir"].numel() > 0:
            emb = self.ir_encoder(batch["ir"])
        else:
            emb = torch.empty(0)
        embeddings.append(emb)

        # Thermal
        if batch["thermal"].numel() > 0:
            emb = self.thermal_encoder(batch["thermal"])
        else:
            emb = torch.empty(0)
        embeddings.append(emb)

        # Build flags tensor: [has_imu, has_radar, has_skeleton, has_depth, has_ir, has_thermal]
        flag_tensor = torch.stack([
            flags["has_imu"], flags["has_radar"], flags["has_skeleton"],
            flags["has_depth"], flags["has_ir"], flags["has_thermal"]
        ], dim=-1)

        # Fusion — request penultimate features for aux head
        need_aux = self.config.flags.use_aux_category_loss
        logits, penultimate = self.fusion(embeddings, flag_tensor,
                                          return_penultimate=need_aux)

        # Auxiliary category classifier
        aux_logits = None
        if need_aux and penultimate is not None:
            aux_logits = self.aux_head(penultimate)

        output = {"logits": logits}
        if aux_logits is not None:
            output["aux_logits"] = aux_logits

        if return_embeddings:
            output["embeddings"] = embeddings

        return output

    def get_parameter_count(self):
        """Return total number of trainable parameters in millions."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad) / 1e6

    def get_parameter_breakdown(self):
        """Per-component trainable parameter counts, for understanding
        where the model's size (and the competition's 100MB budget) is
        actually spent.

        Returns:
            dict of component_name -> {"params_m": float, "mb": float,
            "pct": float of total}, plus a "total" entry. mb assumes
            float32 (4 bytes/param), matching how the competition's size
            limit is normally measured.
        """
        components = {
            "imu_encoder": self.imu_encoder,
            "radar_encoder": self.radar_encoder,
            "skeleton_encoder": self.skeleton_encoder,
            "depth_encoder": self.depth_encoder,
            "ir_encoder": self.ir_encoder,
            "thermal_encoder": self.thermal_encoder,
            "fusion": self.fusion,
            "aux_head": self.aux_head,
        }
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        breakdown = {}
        for name, module in components.items():
            n = sum(p.numel() for p in module.parameters() if p.requires_grad)
            breakdown[name] = {
                "params_m": n / 1e6,
                "mb": n * 4 / 1e6,
                "pct": 100.0 * n / total_params if total_params > 0 else 0.0,
            }
        breakdown["total"] = {
            "params_m": total_params / 1e6,
            "mb": total_params * 4 / 1e6,
            "pct": 100.0,
        }
        return breakdown

    def log_parameter_breakdown(self):
        """Print a formatted per-component parameter/size breakdown."""
        breakdown = self.get_parameter_breakdown()
        active_flags = [
            name for name, on in [
                ("segment_pooling", self.config.flags.use_segment_pooling),
                ("cross_modal_attention", self.config.flags.use_cross_modal_attention),
                ("synthesized_features", self.config.flags.use_synthesized_features),
            ] if on
        ]
        flags_str = ", ".join(active_flags) if active_flags else "none"
        print(f"{'='*60}")
        print(f"  Model size breakdown (float32)")
        print(f"  Active architecture flags: {flags_str}")
        print(f"{'='*60}")
        for name, info in breakdown.items():
            if name == "total":
                continue
            print(f"  {name:16s} {info['params_m']:7.3f}M params  "
                  f"{info['mb']:7.2f} MB  ({info['pct']:5.1f}%)")
        print(f"  {'-'*56}")
        total = breakdown["total"]
        print(f"  {'TOTAL':16s} {total['params_m']:7.3f}M params  "
              f"{total['mb']:7.2f} MB  (budget: {total['mb']:.1f} / 100 MB)")
        print(f"{'='*60}")
