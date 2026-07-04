"""
# CONVENTION: primary — Propagating-exception convention.

HARDataset: loads multimodal clips from the CUHK-X directory structure.

Directory structure (training):
    HAR/data/<modality>/<action>/<user>/<trial>/<files>

Label extraction: the <action> folder name encodes the label.
    e.g., "0_Wash_face" → action_id = 0

Handles missing modalities gracefully (returns fixed-shape zero tensors).
"""
import os
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import Dataset

# Import action categories for auxiliary loss
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import ACTION_CATEGORIES

# Upper-bound constants
MAX_CLIPS = 50000
MAX_FRAMES_PER_CLIP = 4096
MAX_IMU_SAMPLES = 50000
MAX_RADAR_FRAMES = 4096
MAX_SKEL_FRAMES = 4096

# Left/right joint swap pairs for handedness-flip mirroring, verified from
# this codebase's own angle_joints convention below (NOT generic H36M
# literature — this pipeline's own labeling has 11=right_shoulder /
# 14=left_shoulder, the reverse of the "standard" H36M convention, but
# internal consistency is what a mirror-swap needs). Joints 0 (hip/pelvis
# center), 7/8/9/10 (spine/neck/head chain) have no left/right counterpart
# and are left in place.
_SKEL_LR_SWAP_PAIRS = [(1, 4), (2, 5), (3, 6), (11, 14), (12, 15), (13, 16)]


def _skeleton_mirror_permutation():
    perm = list(range(17))
    for a, b in _SKEL_LR_SWAP_PAIRS:
        perm[a], perm[b] = perm[b], perm[a]
    return perm


SKEL_MIRROR_PERM = _skeleton_mirror_permutation()


def parse_user_id(user_folder_name):
    """Extract numeric user ID from folder name like 'user10'.

    Args:
        user_folder_name: str, e.g. "user10" or "user3".

    Returns:
        int user ID, or None if parsing fails.
    """
    try:
        if user_folder_name.startswith("user"):
            return int(user_folder_name[4:])
        return None
    except (ValueError, IndexError):
        return None


def parse_action_label(action_folder_name):
    """Extract action_id from folder name like '0_Wash_face'.

    Args:
        action_folder_name: str, e.g. "0_Wash_face" or "23_Listen_to_music".

    Returns:
        int action_id, or None if parsing fails.
    """
    try:
        parts = action_folder_name.split("_", 1)
        return int(parts[0])
    except (ValueError, IndexError):
        return None




def discover_clips(data_root):
    """Traverse the training data directory and build a clip inventory.

    Each clip is identified by (modality, action_folder, user, trial).

    Args:
        data_root: Path to HAR/data/ directory.

    Returns:
        clips: dict mapping (user, trial) → {modality: file_list}.
        labels: dict mapping (user, trial) → action_id.
    """
    clips = defaultdict(lambda: defaultdict(list))
    labels = {}

    for modality_dir in sorted(data_root.iterdir()):
        if not modality_dir.is_dir():
            continue
        modality = modality_dir.name

        for action_dir in sorted(modality_dir.iterdir()):
            if not action_dir.is_dir():
                continue
            action_id = parse_action_label(action_dir.name)
            if action_id is None:
                continue

            for user_dir in sorted(action_dir.iterdir()):
                if not user_dir.is_dir():
                    continue
                user = user_dir.name

                for trial_dir in sorted(user_dir.iterdir()):
                    if not trial_dir.is_dir():
                        continue
                    trial = trial_dir.name
                    key = (user, trial)

                    # Collect files — search recursively for Skeleton (in predictions/)
                    files = sorted(
                        [str(f) for f in trial_dir.rglob("*") if f.is_file()]
                    )
                    if files:
                        clips[key][modality] = files
                    elif modality == "Skeleton":
                        # Skeleton: also check for predictions/ subfolder
                        pred_dir = trial_dir / "predictions"
                        if pred_dir.is_dir():
                            json_files = sorted(
                                [str(f) for f in pred_dir.glob("*.json")]
                            )
                            if json_files:
                                clips[key][modality] = json_files
                    if key not in labels:
                        labels[key] = action_id

    return dict(clips), labels


def build_clip_list(clips, labels):
    """Convert clip dict into a flat list of (user_id, user, trial, action_id) tuples.

    Args:
        clips: dict from discover_clips().
        labels: dict from discover_clips().

    Returns:
        list of (user_id, user, trial, action_id) tuples.
    """
    clip_list = []
    for (user, trial), action_id in labels.items():
        user_id = parse_user_id(user)
        clip_list.append((user_id, user, trial, action_id))
    return clip_list


class HARDataset(Dataset):
    """PyTorch Dataset for CUHK-X multimodal action recognition.

    Loads up to 6 modalities per clip. Missing modalities return fixed-shape zero tensors.

    Args:
        clips: dict mapping (user, trial) → {modality: file_list}.
        labels: dict mapping (user, trial) → action_id.
        clip_list: list of (user_id, user, trial, action_id) for this split.
        config: Config object with data settings.
        is_train: whether this is the training split (enables augmentations).
    """

    def __init__(self, clips, labels, clip_list, config, is_train=True):
        self.clips = clips
        self.labels = labels
        self.clip_list = clip_list
        self.config = config
        self.is_train = is_train

    def __len__(self):
        return len(self.clip_list)

    def __getitem__(self, idx):
        user_id, user, trial, action_id = self.clip_list[idx]
        key = (user, trial)
        modalities = self.clips.get(key, {})

        sample = {
            "user_id": user_id,
            "user": user,
            "trial": trial,
            "label": torch.tensor(action_id, dtype=torch.long),
            "category_label": torch.tensor(
                ACTION_CATEGORIES.get(action_id, 0), dtype=torch.long),
        }

        # Compute spatial attention mask from raw skeleton bbox (before normalization)
        spatial_mask = None
        if (self.config.flags.use_skeleton_attention_mask
                and self.is_train
                and "Skeleton" in modalities):
            spatial_mask = self._compute_skeleton_mask(modalities)

        # Handedness-flip augmentation: one random decision per sample,
        # applied consistently across every modality (CMI 1st place's
        # "handedness normalization" pattern, ported from a deterministic
        # per-subject correction into a random on-the-fly augmentation).
        # Train-only, same convention as spatial_crop/random_erase below.
        do_flip = (
            self.is_train
            and self.config.flags.use_handedness_flip
            and np.random.rand() < self.config.flags.handedness_flip_p
        )

        # Load each modality. Loaders always return a fixed-shape zero
        # placeholder when a modality is missing or fails to parse — never a
        # truly empty tensor — so every sample in a batch has identical shape
        # per modality and collate_fn can stack without dropping rows (which
        # would desync the batch from labels/flags).
        sample["imu"] = self._load_imu(modalities, mirror=do_flip)
        sample["radar"] = self._load_radar(modalities, mirror=do_flip)
        sample["skeleton"] = self._load_skeleton(modalities, mirror=do_flip)
        sample["depth_color"] = self._load_frames(modalities, "Depth_Color", spatial_mask, mirror=do_flip)
        sample["ir"] = self._load_frames(modalities, "IR", spatial_mask, mirror=do_flip)
        sample["thermal"] = self._load_frames(modalities, "Thermal", spatial_mask, mirror=do_flip)

        # Modality presence flags: based on directory presence, not tensor
        # emptiness (loaders never return empty tensors — see above).
        sample["has_imu"] = torch.tensor("IMU" in modalities, dtype=torch.float32)
        sample["has_radar"] = torch.tensor("Radar" in modalities, dtype=torch.float32)
        sample["has_skeleton"] = torch.tensor("Skeleton" in modalities, dtype=torch.float32)
        sample["has_depth"] = torch.tensor("Depth_Color" in modalities, dtype=torch.float32)
        sample["has_ir"] = torch.tensor("IR" in modalities, dtype=torch.float32)
        sample["has_thermal"] = torch.tensor("Thermal" in modalities, dtype=torch.float32)

        return sample

    def _compute_skeleton_mask(self, modalities):
        """Compute a spatial attention mask from skeleton joint bounding box.

        Loads raw (unnormalized) skeleton joints, computes a 2D bbox,
        and creates a soft mask: 1.0 inside the person bbox, bg_weight outside.

        The mask is resized to frame_size × frame_size.

        Args:
            modalities: dict of modality → file list.

        Returns:
            (num_frames, 1, frame_size, frame_size) float tensor, or None.
        """
        import json
        try:
            skel_root = Path(modalities["Skeleton"][0]).parent
            pred_dir = skel_root / "predictions"
            if pred_dir.exists():
                json_files = sorted(pred_dir.glob("*.json"))
            else:
                json_files = sorted([f for f in modalities["Skeleton"]
                                     if f.endswith(".json")])

            if not json_files:
                return None

            # Collect bbox across all frames
            all_x, all_y = [], []
            all_kps = []
            for jf in json_files:
                with open(jf) as f:
                    data = json.load(f)
                if isinstance(data, list) and len(data) > 0:
                    kps = np.array(data[0].get("keypoints", []), dtype=np.float32)
                    if kps.shape == (17, 3):
                        all_kps.append(kps)
                        valid = kps[:, 2] > 0.1  # confidence threshold
                        if valid.sum() > 3:
                            all_x.extend(kps[valid, 0])
                            all_y.extend(kps[valid, 1])

            if not all_x:
                return None

            # Compute bbox with margin
            margin = 0.15
            x_min = np.percentile(all_x, 5) - margin
            x_max = np.percentile(all_x, 95) + margin
            y_min = np.percentile(all_y, 5) - margin
            y_max = np.percentile(all_y, 95) + margin

            F = self.config.frame_size

            # Create per-frame mask matching closest skeleton frame to each frame index
            # For Depth/IR: 42 skeleton frames map 1:1 to 42 depth frames
            # For Thermal: sample from 42 skeleton frames to match 32 thermal samples
            n_skel = len(all_kps)
            masks = []

            # Use the global bbox for simplicity (per-frame bbox is noisy)
            # Map bbox from skeleton coordinate space to image space
            # Skeleton x,y range is roughly [-1, 1] meters; image is 640×480
            # We approximate: center the bbox and use fixed padding

            mask = np.ones((F, F), dtype=np.float32) * self.config.flags.mask_bg_weight

            # Map x (lateral) to image columns, y (vertical) to image rows
            # Assuming skeleton x maps to width, y maps to height
            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2
            x_span = max(x_max - x_min, 0.3)
            y_span = max(y_max - y_min, 0.3)

            # Normalize to [0,1] then scale to pixel coords
            # Use fixed mapping: skeleton coord range ~[-1.5, 1.5] → [0, F]
            def skel_to_pixel(val, center, span):
                return int(np.clip((val - (center - span * 0.7)) / (span * 1.4) * F, 0, F - 1))

            col_start = skel_to_pixel(x_min, x_center, x_span)
            col_end   = skel_to_pixel(x_max, x_center, x_span)
            row_start = skel_to_pixel(y_min, y_center, y_span)
            row_end   = skel_to_pixel(y_max, y_center, y_span)

            if col_end > col_start and row_end > row_start:
                mask[row_start:row_end, col_start:col_end] = 1.0

            # Apply Gaussian blur for soft transition
            from scipy.ndimage import gaussian_filter
            mask = gaussian_filter(mask, sigma=F / 40.0)
            mask = np.clip(mask, self.config.flags.mask_bg_weight, 1.0)

            mask_tensor = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0)  # (1, 1, F, F)
            return mask_tensor
        except Exception:
            return None

    # ---- Real data loaders (EDA-informed) ----

    def _load_imu(self, modalities, mirror=False):
        """Load and preprocess IMU CSV data from 2 files (5 body sensors).

        CUHK-X IMU: 2 CSVs per trial.
          down(LL+RL).csv — Left Leg + Right Leg (2 WitMotion sensors)
          up(LA+RA+C).csv  — Left Arm + Right Arm + Chest (3 sensors)

        Features extracted per sensor (19 dims):
          acc_ms2(3), rot_6d(6), gyro_rad(3), linear_acc(3), jerk(3), acc_mag(1)

        Args:
            modalities: dict of modality → file list.
            mirror: if True, apply handedness-flip mirroring (see
                preprocessing.imu_utils.process_imu_trial).

        Returns:
            (imu_seq_len, 95) float tensor (19 feat × 5 sensors),
            or zero-filled tensor of the same shape if IMU missing.
        """
        if "IMU" not in modalities:
            return torch.zeros(self.config.imu_seq_len, self.config.imu_input_dim,
                               dtype=torch.float32)

        from preprocessing.imu_utils import process_imu_trial

        try:
            return process_imu_trial(
                modalities["IMU"],
                target_seq_len=self.config.imu_seq_len,
                time_delta=1.0 / self.config.imu_target_hz,
                use_synthesized=self.config.flags.use_synthesized_features,
                mirror=mirror,
            )
        except Exception:
            return torch.zeros(self.config.imu_seq_len, self.config.imu_input_dim,
                               dtype=torch.float32)

    def _load_radar(self, modalities, mirror=False):
        """Load radar point cloud CSV.

        CUHK-X Radar: 1 CSV per trial, ~82 frames, ~267 detections.
        Columns: timestamp, frame, DetObj#, x, y, z, v, snr, noise

        Groups by frame, pads each frame to max_points (64).

        Args:
            modalities: dict of modality → file list.
            mirror: if True, negate the lateral x-coordinate (handedness
                flip). Doppler velocity v is a radial (line-of-sight)
                scalar, invariant under this reflection — |point| is
                unchanged since (-x)^2 = x^2 — so only x is negated.

        Returns:
            (radar_seq_len, max_points, 6) float tensor,
            or zero-filled tensor of the same shape if Radar missing.
        """
        import pandas as pd
        if "Radar" not in modalities:
            return torch.zeros(self.config.radar_seq_len, self.config.radar_max_points,
                               self.config.radar_point_dim, dtype=torch.float32)

        try:
            radar_path = modalities["Radar"][0]
            df = pd.read_csv(radar_path)
            # Features: x, y, z, v, snr, noise (ignore DetObj#)
            feat_cols = ["x", "y", "z", "v", "snr", "noise"]

            frames = []
            max_pts = self.config.radar_max_points
            max_frames = self.config.radar_seq_len

            for frame_id, grp in df.groupby("frame"):
                pts = grp[feat_cols].values.astype(np.float32)
                if mirror:
                    pts = pts.copy()
                    pts[:, 0] = -pts[:, 0]
                # Pad/truncate to max_points
                if pts.shape[0] < max_pts:
                    pad = np.zeros((max_pts - pts.shape[0], len(feat_cols)),
                                   dtype=np.float32)
                    pts = np.concatenate([pts, pad], axis=0)
                else:
                    pts = pts[:max_pts]
                frames.append(pts)
                if len(frames) >= max_frames:
                    break

            # Pad frames to max_frames
            while len(frames) < max_frames:
                frames.append(np.zeros((max_pts, len(feat_cols)), dtype=np.float32))

            result = np.stack(frames[:max_frames], axis=0)  # (F, P, D)
            return torch.from_numpy(result)
        except Exception:
            return torch.zeros(self.config.radar_seq_len, self.config.radar_max_points,
                               self.config.radar_point_dim, dtype=torch.float32)

    def _load_skeleton(self, modalities, mirror=False):
        """Load skeleton keypoints and compute engineered features.

        CUHK-X Skeleton: ~42 JSONs in predictions/ subfolder.
        Each JSON: [{"keypoints": [[x,y,z]*17], "keypoint_scores": [s*17]}]

        Engineered features (119 dims total):
          - Joint positions (51): 17 joints × 3D, hip-centered, spine-normalized
          - Joint velocities (51): per-joint Δpos/Δt (CMI 3rd place pattern)
          - Bone angles (17): angle at each joint from connected bones

        Args:
            modalities: dict of modality → file list.
            mirror: if True, apply handedness-flip mirroring: negate the
                lateral x-coordinate and swap left/right joint pairs
                (SKEL_MIRROR_PERM) before any hip-centering/normalization,
                so the same downstream code (including the fixed
                angle_joints indices below) operates on a physically
                valid reflected pose.

        Returns:
            (skel_seq_len, 119) float tensor,
            or zero-filled tensor of the same shape if Skeleton missing.
        """
        import json
        if "Skeleton" not in modalities:
            return self._skeleton_fallback()

        try:
            # Find predictions/ directory
            skel_root = Path(modalities["Skeleton"][0]).parent
            pred_dir = skel_root / "predictions"
            if not pred_dir.exists():
                # Files might be directly in the trial directory or in a subfolder
                json_files = sorted([f for f in modalities["Skeleton"]
                                     if f.endswith(".json")])
                if not json_files:
                    return torch.empty(0)
                # Load from file list directly
                keypoints_list = []
                for jf in json_files:
                    with open(jf) as f:
                        data = json.load(f)
                    if isinstance(data, list) and len(data) > 0:
                        kps = np.array(data[0].get("keypoints", []), dtype=np.float32)
                        if kps.shape == (17, 3):
                            keypoints_list.append(kps)
            else:
                json_files = sorted(pred_dir.glob("*.json"))
                keypoints_list = []
                for jf in json_files:
                    with open(jf) as f:
                        data = json.load(f)
                    if isinstance(data, list) and len(data) > 0:
                        kps = np.array(data[0].get("keypoints", []), dtype=np.float32)
                        if kps.shape == (17, 3):
                            keypoints_list.append(kps)

            if not keypoints_list:
                return self._skeleton_fallback()

            kps_array = np.stack(keypoints_list, axis=0)  # (T, 17, 3)

            if mirror:
                kps_array = kps_array[:, SKEL_MIRROR_PERM, :].copy()
                kps_array[:, :, 0] = -kps_array[:, :, 0]

            # Normalize: center on hip (joint 0), scale by spine length
            hip = kps_array[:, 0:1, :]  # (T, 1, 3)
            kps_array = kps_array - hip
            spine = kps_array[:, 7, :]
            spine_len = np.linalg.norm(spine, axis=-1, keepdims=True)
            spine_len = np.where(spine_len < 1e-6, 1.0, spine_len)
            kps_array = kps_array / spine_len[:, :, None]

            # --- Raw-only mode: positions only, 51 dims ---
            if not self.config.flags.use_synthesized_features:
                flat = kps_array.reshape(kps_array.shape[0], -1).astype(np.float32)
                return self._skel_pad_trunc(flat)

            # --- Synthesized features: velocities + bone angles → 119 dims ---
            # Joint velocities (Δpos/Δt, 10fps → 0.1s)
            vel = np.zeros_like(kps_array)
            vel[1:] = (kps_array[1:] - kps_array[:-1]) / 0.1
            vel = np.nan_to_num(vel, nan=0.0)

            # Bone angles at key joints (cosine of angle between connected bones)
            angle_joints = [
                (0, 1, 2),     # right_hip angle (hip→right_hip→right_knee)
                (0, 4, 5),     # left_hip angle
                (1, 2, 3),     # right_knee angle
                (4, 5, 6),     # left_knee angle
                (7, 8, 9),     # neck angle
                (7, 11, 12),   # right_shoulder angle
                (7, 14, 15),   # left_shoulder angle
                (11, 12, 13),  # right_elbow angle
                (14, 15, 16),  # left_elbow angle
                (2, 1, 0),     # right_hip flexion
                (5, 4, 0),     # left_hip flexion
                (8, 7, 0),     # spine tilt
            ]

            bone_angles = np.zeros((kps_array.shape[0], 17), dtype=np.float32)
            for ji, (j0, j1, j2) in enumerate(angle_joints):
                v1 = kps_array[:, j0] - kps_array[:, j1]  # (T, 3)
                v2 = kps_array[:, j2] - kps_array[:, j1]  # (T, 3)
                # Cosine of angle = dot(v1,v2) / (|v1||v2|)
                dot = np.sum(v1 * v2, axis=-1)
                norm = np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1)
                norm = np.where(norm < 1e-8, 1.0, norm)
                cos_angle = np.clip(dot / norm, -1.0, 1.0)
                bone_angles[:, ji] = cos_angle

            # Concatenate: positions(51) + velocities(51) + bone_angles(12→pad to 17)
            pos_flat = kps_array.reshape(kps_array.shape[0], -1).astype(np.float32)  # (T, 51)
            vel_flat = vel.reshape(vel.shape[0], -1).astype(np.float32)              # (T, 51)
            ang_flat = np.pad(bone_angles, ((0, 0), (0, 17 - bone_angles.shape[1])),
                              constant_values=0.0)  # (T, 17)
            flat = np.concatenate([pos_flat, vel_flat, ang_flat], axis=-1)  # (T, 119)
            return self._skel_pad_trunc(flat)
        except Exception:
            return self._skeleton_fallback()

    def _skeleton_fallback(self):
        """Return a consistent zero skeleton tensor of expected dims."""
        return torch.zeros(self.config.skel_seq_len,
                           self._skeleton_output_dim(),
                           dtype=torch.float32)

    def _skel_pad_trunc(self, flat):
        """Pad/truncate skeleton time dimension to skel_seq_len."""
        target = self.config.skel_seq_len
        if flat.shape[0] < target:
            pad = np.zeros((target - flat.shape[0], flat.shape[1]), dtype=np.float32)
            flat = np.concatenate([flat, pad], axis=0)
        else:
            flat = flat[:target]
        return torch.from_numpy(flat)

    def _skeleton_output_dim(self):
        """Expected skeleton feature dimension for current config."""
        return 119 if self.config.flags.use_synthesized_features else 51

    def _load_frames(self, modalities, modality_name, spatial_mask=None, mirror=False):
        """Load and preprocess frame images for Depth_Color, IR, or Thermal.

        CUHK-X frame modalities:
          Depth_Color: 640×480 RGB PNG (~42 frames at 10fps)
          IR:          640×480 grayscale PNG (~42 frames at 10fps)
          Thermal:     320×240 RGB JPG (~118 frames at 25fps)

        Applies (if enabled via config.flags):
          - Skeleton-guided spatial attention mask (suppress background)
          - Random spatial crop + resize (background jitter)
          - RandomErase (patch dropout)

        Args:
            modalities: dict of modality → file list.
            modality_name: "Depth_Color", "IR", or "Thermal".
            spatial_mask: (1, 1, F, F) float attention mask, or None.
            mirror: if True, apply handedness-flip mirroring: horizontal
                (left-right) flip of each frame. spatial_mask is flipped
                the same way (once, up front) so it stays aligned with
                the now-mirrored frame content — the mask was computed
                from the un-mirrored skeleton bbox, so it must be flipped
                to match.

        Returns:
            (N_frames, C, H, W) float tensor. C=3 for RGB, C=1 for IR.
            Returns a zero-filled tensor of the same shape if modality missing.
        """
        from PIL import Image

        if mirror and spatial_mask is not None:
            spatial_mask = torch.flip(spatial_mask, dims=[-1])

        # Determine in_channels and frame count (needed for both the
        # missing-modality placeholder and the real loading path)
        if modality_name == "IR":
            in_channels = 1
            num_sample = self.config.num_frames
        elif modality_name == "Thermal":
            in_channels = 3
            num_sample = self.config.num_thermal_frames
        else:  # Depth_Color
            in_channels = 3
            num_sample = self.config.num_frames

        if modality_name not in modalities:
            return torch.zeros(num_sample, in_channels, self.config.frame_size,
                               self.config.frame_size, dtype=torch.float32)

        try:
            files = sorted(modalities[modality_name])
            if not files:
                return torch.zeros(num_sample, in_channels, self.config.frame_size,
                                   self.config.frame_size, dtype=torch.float32)

            size = self.config.frame_size
            flags = self.config.flags

            # Uniform sampling
            n_total = len(files)
            if n_total <= num_sample:
                indices = list(range(n_total))
            else:
                step = n_total / num_sample
                indices = [int(i * step) for i in range(num_sample)]

            # --- Spatial crop parameters (computed once per clip) ---
            crop_on = self.is_train and flags.use_spatial_crop
            if crop_on:
                crop_scale = np.random.uniform(*flags.spatial_crop_scale)
                crop_size = int(size * crop_scale)
                max_offset = size - crop_size
                offset_y = np.random.randint(0, max_offset + 1) if max_offset > 0 else 0
                offset_x = np.random.randint(0, max_offset + 1) if max_offset > 0 else 0

            # --- RandomErase parameters (computed once per clip) ---
            erase_on = self.is_train and flags.use_random_erase
            if erase_on:
                erase_h = int(np.random.uniform(*flags.erase_scale) * size)
                erase_w = int(erase_h * np.random.uniform(*flags.erase_ratio))
                erase_w = min(erase_w, size - 1)
                erase_h = min(erase_h, size - 1)
                erase_y = np.random.randint(0, max(1, size - erase_h))
                erase_x = np.random.randint(0, max(1, size - erase_w))

            frames = []
            for i in indices:
                img = Image.open(files[i])
                if modality_name == "IR":
                    img = img.convert("L")
                else:
                    img = img.convert("RGB")
                img = img.resize((size, size), Image.BILINEAR)
                arr = np.array(img, dtype=np.float32) / 255.0

                if modality_name == "IR":
                    arr = arr[np.newaxis, :, :]  # (1, H, W)
                else:
                    arr = arr.transpose(2, 0, 1)  # (3, H, W)

                # --- Apply handedness-flip mirroring (before mask/crop/erase,
                # which then all operate consistently in the flipped space) ---
                if mirror:
                    arr = arr[:, :, ::-1].copy()

                # --- Apply skeleton-guided spatial attention mask ---
                if spatial_mask is not None:
                    # spatial_mask is (1, 1, F, F), broadcast to (C, H, W)
                    mask_np = spatial_mask.squeeze(0).numpy()  # (1, F, F)
                    arr = arr * mask_np

                # --- Apply random spatial crop ---
                if crop_on:
                    cropped = arr[:, offset_y:offset_y + crop_size,
                                    offset_x:offset_x + crop_size]
                    # Resize back via numpy (simple, no PIL overhead)
                    # Use repeat+reshape for simple nearest-neighbor upscale
                    arr = np.zeros_like(arr)
                    for c in range(arr.shape[0]):
                        from PIL import Image as PILImage
                        ch_img = PILImage.fromarray(
                            (cropped[c] * 255).astype(np.uint8))
                        ch_img = ch_img.resize((size, size), PILImage.BILINEAR)
                        arr[c] = np.array(ch_img, dtype=np.float32) / 255.0

                # --- Apply RandomErase ---
                if erase_on:
                    arr[:, erase_y:erase_y + erase_h,
                           erase_x:erase_x + erase_w] = 0.0

                frames.append(arr)

            # Pad if needed
            while len(frames) < num_sample:
                frames.append(np.zeros_like(frames[0]))

            result = np.stack(frames, axis=0)  # (N, C, H, W)
            return torch.from_numpy(result)
        except Exception:
            c = 1 if modality_name == "IR" else 3
            num = (self.config.num_thermal_frames if modality_name == "Thermal"
                   else self.config.num_frames)
            return torch.zeros(num, c, self.config.frame_size, self.config.frame_size,
                               dtype=torch.float32)
