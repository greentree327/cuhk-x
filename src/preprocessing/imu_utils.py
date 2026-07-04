"""
# CONVENTION: secondary — Return-sentinel convention.
# Functions return None on failure for boundary interoperability.
# CONVENTION: primary — Propagating-exception convention for internal functions.

IMU preprocessing utilities ported from CMI 2nd place solution.
- Quaternion normalization
- Quaternion → 6D rotation representation
- Gravity removal from acceleration
- Angular velocity from quaternion differences
"""
import numpy as np
from scipy.spatial.transform import Rotation as R
import torch

# Features per sensor after compute_imu_features (27: acc(3)+rot6d(6)+gyro(3)+
# lin_acc(3)+jerk(3)+acc_mag(1)+gyro_mag(1)+lin_acc_mag(1)+jerk_mag(1)+
# angular_jerk(3)+acc_roll_std(1)+gyro_roll_std(1))
IMU_FEAT_PER_SENSOR = 27
IMU_NUM_SENSORS = 5
IMU_TOTAL_FEAT = IMU_FEAT_PER_SENSOR * IMU_NUM_SENSORS  # 135

# Device-name prefixes (before the trailing "(MAC-address)" suffix), in the
# same alphabetical order pandas' groupby(COL_DEVICE) naturally produces —
# verified against real trial data: WTC < WTLA < WTLL < WTRA < WTRL.
CANONICAL_SENSOR_ORDER = ["WTC", "WTLA", "WTLL", "WTRA", "WTRL"]
# Left/right sensor swap for handedness-flip mirroring. Chest has no
# left/right counterpart and maps to itself (omitted -> unchanged via .get).
SENSOR_LR_SWAP = {"WTLA": "WTRA", "WTRA": "WTLA", "WTLL": "WTRL", "WTRL": "WTLL"}

# Sentinel for functions returning Optional values
_SENTINEL = object()


def normalize_quaternion(quat):
    """Normalize quaternion to unit length.

    Args:
        quat: (..., 4) array in [x, y, z, w] format.

    Returns:
        Normalized quaternion, same shape.
        Returns original if norm is near-zero (guard: division by zero).
    """
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    norm = np.where(norm > 1e-8, norm, 1.0)
    return quat / norm


def quaternion_to_6d_rotation(quat):
    """Convert quaternion to continuous 6D rotation representation.

    The 6D representation uses the first two columns of the 3×3 rotation
    matrix, avoiding the discontinuity of quaternion sign ambiguity.

    Args:
        quat: (..., 4) array in [x, y, z, w] format.

    Returns:
        (..., 6) array of 6D rotation vectors. NaN entries where input
        is invalid or conversion fails.
    """
    if quat.ndim == 1:
        quat = quat.reshape(1, -1)

    has_nan = np.any(np.isnan(quat), axis=-1)
    result = np.full((*quat.shape[:-1], 6), np.nan)

    valid_mask = ~has_nan & ~np.all(np.isclose(quat, 0), axis=-1)
    if not np.any(valid_mask):
        return result

    valid_quat = quat[valid_mask]

    try:
        valid_quat_norm = normalize_quaternion(valid_quat)
        rotations = R.from_quat(valid_quat_norm)
        rotation_matrices = rotations.as_matrix()
        # First two columns = 6D representation
        result[valid_mask] = rotation_matrices[:, :, :2].reshape(-1, 6)
    except (ValueError, RuntimeError):
        pass

    return result


def remove_gravity_from_acc(acc_data, rot_data, gravity_world=np.array([0, 0, 9.81])):
    """Remove gravity component from acceleration using quaternion orientation.

    Transforms world-frame gravity into sensor frame via inverse rotation,
    then subtracts from measured acceleration.

    Args:
        acc_data: (N, 3) array [acc_x, acc_y, acc_z] in m/s².
        rot_data: (N, 4) quaternion array [x, y, z, w].
        gravity_world: (3,) gravity vector in world frame.

    Returns:
        (N, 3) linear acceleration array. NaN where computation fails.
    """
    acc_values = np.asarray(acc_data)
    quat_values = np.asarray(rot_data)
    num_samples = acc_values.shape[0]
    linear_accel = np.full_like(acc_values, np.nan)

    for i in range(num_samples):
        if np.any(np.isnan(acc_values[i])) or np.any(np.isnan(quat_values[i])):
            continue
        if np.all(np.isclose(quat_values[i], 0)):
            linear_accel[i, :] = acc_values[i, :]
            continue

        try:
            quat_norm = normalize_quaternion(quat_values[i:i + 1])[0]
            rotation = R.from_quat(quat_norm)
            gravity_sensor_frame = rotation.apply(gravity_world, inverse=True)
            linear_accel[i, :] = acc_values[i, :] - gravity_sensor_frame
        except (ValueError, RuntimeError):
            continue

    return linear_accel


def calculate_angular_velocity_from_quat(rot_data, time_delta=1 / 200):
    """Compute angular velocity from sequential quaternions.

    Uses the rotation vector (axis-angle) of the delta rotation between
    consecutive orientations, divided by time_delta.

    Args:
        rot_data: (N, 4) quaternion array [x, y, z, w].
        time_delta: float, time between samples in seconds.

    Returns:
        (N, 3) angular velocity array [wx, wy, wz] in rad/s.
        NaN where computation fails; last entry is always NaN.
    """
    quat_values = np.asarray(rot_data)
    num_samples = quat_values.shape[0]
    angular_vel = np.full((num_samples, 3), np.nan)

    for i in range(num_samples - 1):
        q_t = quat_values[i]
        q_t_plus_dt = quat_values[i + 1]

        if (
            np.any(np.isnan(q_t))
            or np.any(np.isnan(q_t_plus_dt))
            or np.all(np.isclose(q_t, 0))
            or np.all(np.isclose(q_t_plus_dt, 0))
        ):
            continue

        try:
            q_t_norm = normalize_quaternion(q_t.reshape(1, -1))[0]
            q_t_plus_dt_norm = normalize_quaternion(q_t_plus_dt.reshape(1, -1))[0]
            rot_t = R.from_quat(q_t_norm)
            rot_t_plus_dt = R.from_quat(q_t_plus_dt_norm)
            delta_rot = rot_t.inv() * rot_t_plus_dt
            angular_vel[i, :] = delta_rot.as_rotvec() / time_delta
        except (ValueError, RuntimeError):
            continue

    return angular_vel


def _mirror_quaternion_lr(quat_scipy):
    """Mirror an [x,y,z,w] quaternion array through the X=0 plane (a
    left-right body reflection), via rotation-matrix conjugation
    R' = P @ R @ P with P = diag(-1,1,1) — then converted back to a
    quaternion. Naively negating quaternion components does not
    correctly represent a mirrored orientation; this matches CMI 1st
    place's mirror_quaternion (verified against the actual notebook
    code in Similar Competition/1st_place_solution/public_solution_ogurtsov).

    Args:
        quat_scipy: (N, 4) array, scipy [x, y, z, w] order.

    Returns:
        (N, 4) mirrored quaternion, same order. Invalid (NaN/all-zero)
        rows pass through unchanged, matching the defensive per-row
        convention used elsewhere in this file (see
        remove_gravity_from_acc, calculate_angular_velocity_from_quat).
    """
    quat_scipy = np.asarray(quat_scipy, dtype=np.float64)
    num_samples = quat_scipy.shape[0]
    mirrored = quat_scipy.copy()
    P = np.diag([-1.0, 1.0, 1.0])

    for i in range(num_samples):
        row = quat_scipy[i]
        if np.any(np.isnan(row)) or np.all(np.isclose(row, 0)):
            continue
        try:
            q_norm = normalize_quaternion(row.reshape(1, -1))[0]
            rot_mat = R.from_quat(q_norm).as_matrix()
            flipped_mat = P @ rot_mat @ P
            mirrored[i] = R.from_matrix(flipped_mat).as_quat()
        except (ValueError, RuntimeError):
            continue

    return mirrored


def mirror_imu_sensor(acc, gyro, quat):
    """Mirror one IMU sensor's raw readings through a left-right body
    reflection (negate the lateral/X axis), for handedness-flip
    augmentation (CMI 1st place's "handedness normalization" pattern,
    ported and generalized from a deterministic per-subject correction
    into a random on-the-fly augmentation).

    Three physically distinct quantities, three distinct transforms:
      - Acceleration is a true (polar) vector: only its X component
        flips sign under an X-axis reflection.
      - Angular velocity (gyro) is a pseudovector (axial vector), which
        transforms *oppositely* to a polar vector under a reflection P:
        omega' = det(P) * P @ omega. For P = diag(-1,1,1) (det=-1), the
        component ALONG the mirror normal (X) is unchanged, while the
        two in-plane components (Y, Z) flip sign — the reverse of the
        acceleration pattern.
      - Quaternion (orientation) mirrors via rotation-matrix
        conjugation (see _mirror_quaternion_lr), not component
        negation.

    Args:
        acc: (N, 3) raw acceleration, any consistent unit (g or m/s^2 —
            a sign flip doesn't depend on scale).
        gyro: (N, 3) raw angular velocity, any consistent unit.
        quat: (N, 4) quaternion in CUHK-X [w, x, y, z] order.

    Returns:
        (acc_mirrored, gyro_mirrored, quat_mirrored) — same shapes,
        units, and (for quat) column order as the inputs.
    """
    acc = np.asarray(acc, dtype=np.float64).copy()
    gyro = np.asarray(gyro, dtype=np.float64)
    quat = np.asarray(quat, dtype=np.float64)

    acc[:, 0] = -acc[:, 0]

    gyro_mirrored = gyro.copy()
    gyro_mirrored[:, 1] = -gyro[:, 1]
    gyro_mirrored[:, 2] = -gyro[:, 2]
    # gyro_mirrored[:, 0] (X component) intentionally left unchanged.

    # CUHK-X [w,x,y,z] -> scipy [x,y,z,w] (same reordering as compute_imu_features)
    quat_scipy = np.zeros_like(quat)
    quat_scipy[:, 0] = quat[:, 1]
    quat_scipy[:, 1] = quat[:, 2]
    quat_scipy[:, 2] = quat[:, 3]
    quat_scipy[:, 3] = quat[:, 0]

    quat_scipy_mirrored = _mirror_quaternion_lr(quat_scipy)

    quat_mirrored = np.zeros_like(quat)
    quat_mirrored[:, 0] = quat_scipy_mirrored[:, 3]  # w
    quat_mirrored[:, 1] = quat_scipy_mirrored[:, 0]  # x
    quat_mirrored[:, 2] = quat_scipy_mirrored[:, 1]  # y
    quat_mirrored[:, 3] = quat_scipy_mirrored[:, 2]  # z

    return acc, gyro_mirrored, quat_mirrored


def compute_imu_features(acc, quat, gyro=None, time_delta=1 / 20,
                         use_synthesized=True):
    """Compute IMU features from raw sensor data.

    Raw-only mode (use_synthesized=False): 10 features per sensor
      acc_x/y/z (m/s²), gyro_x/y/z (rad/s), quat_w/x/y/z

    Full mode (use_synthesized=True): 27 features per sensor
      Above + rot_6d, linear_acc, jerk, magnitudes, angular_jerk, rolling stats

    Args:
        acc: (N, 3) raw acceleration in g units.
        quat: (N, 4) quaternion [w,x,y,z] (CUHK-X order).
        gyro: (N, 3) angular velocity in °/s, or None.
        time_delta: sample interval in seconds.
        use_synthesized: if False, return raw-only features.

    Returns:
        (N, 10) or (N, 27) float32 array. None if input empty.
    """
    G_TO_MS2 = 9.80665
    DEG_TO_RAD = np.pi / 180.0

    acc = np.asarray(acc, dtype=np.float64)
    quat = np.asarray(quat, dtype=np.float64)

    if acc.shape[0] == 0:
        return None

    # Convert acc from g to m/s²
    acc_ms2 = acc * G_TO_MS2

    # Convert quaternion from [w,x,y,z] (CUHK-X) to [x,y,z,w] (scipy)
    quat_scipy = np.zeros_like(quat)
    quat_scipy[:, 0] = quat[:, 1]  # x
    quat_scipy[:, 1] = quat[:, 2]  # y
    quat_scipy[:, 2] = quat[:, 3]  # z
    quat_scipy[:, 3] = quat[:, 0]  # w

    # 1) 6D rotation from quaternion
    rot_6d = quaternion_to_6d_rotation(quat_scipy)  # (N, 6)

    # 2) Gyro: use raw if available, otherwise compute from quat diff
    if gyro is not None:
        gyro = np.asarray(gyro, dtype=np.float64)
        gyro_rad = gyro * DEG_TO_RAD  # °/s → rad/s
    else:
        gyro_rad = calculate_angular_velocity_from_quat(quat_scipy, time_delta)
    gyro_rad = np.nan_to_num(gyro_rad, nan=0.0)

    # --- Raw-only mode: 10 features per sensor ---
    if not use_synthesized:
        raw_feat = np.concatenate([acc_ms2, gyro_rad, quat], axis=-1)  # (N, 10)
        return np.nan_to_num(raw_feat, nan=0.0).astype(np.float32)

    # --- Synthesized features (only computed if needed) ---

    # 3) Linear acceleration: gravity removed (pass acc in m/s²)
    linear_acc = remove_gravity_from_acc(acc_ms2, quat_scipy)  # (N, 3)
    linear_acc = np.nan_to_num(linear_acc, nan=0.0)

    # 4) Jerk: derivative of linear acceleration
    jerk = np.zeros_like(linear_acc)
    jerk[1:] = np.diff(linear_acc, axis=0) / time_delta

    # 5) Acceleration magnitude
    acc_mag = np.linalg.norm(acc_ms2, axis=-1, keepdims=True)  # (N, 1)

    # 6) Gyro magnitude (CMI 1st place Ogurtsov)
    gyro_mag = np.linalg.norm(gyro_rad, axis=-1, keepdims=True)

    # 7) Linear acc magnitude
    linear_acc_mag = np.linalg.norm(linear_acc, axis=-1, keepdims=True)

    # 8) Jerk magnitude
    jerk_mag = np.linalg.norm(jerk, axis=-1, keepdims=True)

    # 9) Angular jerk — derivative of gyro (CMI 3rd place minerppdy)
    angular_jerk = np.zeros_like(gyro_rad)
    angular_jerk[1:] = np.diff(gyro_rad, axis=0) / time_delta

    # 10) Rolling window statistics (CMI 30th — frequency proxy)
    WINDOW = 5
    acc_roll = np.zeros((acc_ms2.shape[0], 1), dtype=np.float64)
    gyro_roll = np.zeros((gyro_rad.shape[0], 1), dtype=np.float64)
    am = acc_mag[:, 0]; gm = gyro_mag[:, 0]
    for i in range(acc_ms2.shape[0]):
        lo, hi = max(0, i - WINDOW), min(acc_ms2.shape[0], i + WINDOW + 1)
        acc_roll[i, 0] = np.std(am[lo:hi])
        gyro_roll[i, 0] = np.std(gm[lo:hi])

    # Concatenate: 3+6+3+3+3+1 + 1+1+1 + 3 + 1+1 = 27
    features = np.concatenate([
        acc_ms2, rot_6d, gyro_rad,            # 3 + 6 + 3 = 12
        linear_acc, jerk, acc_mag,             # 3 + 3 + 1 = 7
        gyro_mag, linear_acc_mag, jerk_mag,    # 1 + 1 + 1 = 3
        angular_jerk,                          # 3
        acc_roll, gyro_roll,                   # 1 + 1 = 2
    ], axis=-1)  # 12 + 7 + 3 + 3 + 2 = 27

    features = np.nan_to_num(features, nan=0.0).astype(np.float32)
    return features


def process_imu_trial(file_paths, target_seq_len=128, time_delta=1 / 20,
                      use_synthesized=True, mirror=False):
    """Load and preprocess one trial's IMU data (2 CSV files, 5 sensors).

    CUHK-X IMU files:
      down(LL+RL).csv — Left Leg + Right Leg (2 sensors)
      up(LA+RA+C).csv  — Left Arm + Right Arm + Chest (3 sensors)

    Args:
        file_paths: list of Path or str to 2 IMU CSV files.
        target_seq_len: pad/truncate to this length.
        time_delta: nominal sample interval.
        use_synthesized: if False, use raw-only features (10/sensor).
        mirror: if True, apply handedness-flip augmentation — mirror
            each sensor's raw signal (see mirror_imu_sensor) and swap
            the left/right sensor slots (LA<->RA, LL<->RL) so the
            output represents a left-right-reflected body, still laid
            out in the same canonical sensor order as the unmirrored
            path.

    Returns:
        (target_seq_len, D) float32 tensor. D = 50 (raw) or 135 (synthesized).
    """
    import pandas as pd

    COL_TIME = '时间'
    COL_DEVICE = '设备名称'
    COL_ACC = ['加速度X(g)', '加速度Y(g)', '加速度Z(g)']
    COL_GYRO = ['角速度X(°/s)', '角速度Y(°/s)', '角速度Z(°/s)']
    COL_QUAT = ['四元数0()', '四元数1()', '四元数2()', '四元数3()']

    # Load and merge CSVs
    frames = []
    for fp in file_paths:
        df = pd.read_csv(fp)
        df[COL_TIME] = pd.to_datetime(df[COL_TIME])
        frames.append(df)
    merged = pd.concat(frames, ignore_index=True).sort_values(COL_TIME)

    # Process each sensor independently, then concatenate
    feat_per_sensor = 27 if use_synthesized else 10
    total_feat = feat_per_sensor * IMU_NUM_SENSORS  # 135 or 50

    if not mirror:
        sensor_features = []
        for dev_name, grp in merged.groupby(COL_DEVICE):
            acc = grp[COL_ACC].values.astype(np.float64)
            quat = grp[COL_QUAT].values.astype(np.float64)
            gyro = grp[COL_GYRO].values.astype(np.float64)

            feats = compute_imu_features(acc, quat, gyro=gyro, time_delta=time_delta,
                                          use_synthesized=use_synthesized)
            if feats is None:
                feats = np.zeros((0, feat_per_sensor), dtype=np.float32)
            sensor_features.append(feats)
    else:
        # Handedness flip: mirror each sensor's raw signal, then swap the
        # left/right sensor slots (LA<->RA, LL<->RL) so the position that
        # used to hold the left sensor's data now holds the (mirrored)
        # right sensor's data and vice versa — the tensor layout still
        # matches "canonical sensor order", just describing a reflected
        # body. Matched on device-name PREFIX (not positional/alphabetical
        # order), since that's the only thing guaranteed stable across
        # trials — verified against real device names.
        sensor_by_code = {}
        for dev_name, grp in merged.groupby(COL_DEVICE):
            code = dev_name.split("(")[0]
            acc = grp[COL_ACC].values.astype(np.float64)
            quat = grp[COL_QUAT].values.astype(np.float64)
            gyro = grp[COL_GYRO].values.astype(np.float64)

            acc, gyro, quat = mirror_imu_sensor(acc, gyro, quat)
            feats = compute_imu_features(acc, quat, gyro=gyro, time_delta=time_delta,
                                          use_synthesized=use_synthesized)
            if feats is None:
                feats = np.zeros((0, feat_per_sensor), dtype=np.float32)
            sensor_by_code[SENSOR_LR_SWAP.get(code, code)] = feats

        sensor_features = [sensor_by_code[code] for code in CANONICAL_SENSOR_ORDER
                           if code in sensor_by_code]

    if not sensor_features:
        return torch.zeros(target_seq_len, total_feat, dtype=torch.float32)

    # Align to same length: use the longest sensor, pad others with zeros
    max_len = max(f.shape[0] for f in sensor_features)
    aligned = []
    for f in sensor_features:
        if f.shape[0] < max_len:
            pad = np.zeros((max_len - f.shape[0], feat_per_sensor), dtype=np.float32)
            f = np.concatenate([f, pad], axis=0)
        aligned.append(f)

    # Concatenate across sensors
    combined = np.concatenate(aligned, axis=-1)

    # Handle variable sensor count
    num_sensors = combined.shape[1] // feat_per_sensor
    if num_sensors < IMU_NUM_SENSORS:
        missing = np.zeros((combined.shape[0],
                            (IMU_NUM_SENSORS - num_sensors) * feat_per_sensor),
                           dtype=np.float32)
        combined = np.concatenate([combined, missing], axis=-1)
    elif num_sensors > IMU_NUM_SENSORS:
        combined = combined[:, :total_feat]

    # Pad/truncate time
    if combined.shape[0] < target_seq_len:
        pad = np.zeros((target_seq_len - combined.shape[0], total_feat),
                       dtype=np.float32)
        combined = np.concatenate([combined, pad], axis=0)
    else:
        combined = combined[:target_seq_len]

    return torch.from_numpy(combined)


# Pre-allocated upper-bound constant for feature concatenation loop contexts
MAX_IMU_FEATURES = 128
