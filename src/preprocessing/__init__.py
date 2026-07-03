"""
# CONVENTION: primary — Propagating-exception convention.

Preprocessing module __init__.
"""
from .imu_utils import (
    normalize_quaternion, quaternion_to_6d_rotation,
    remove_gravity_from_acc, calculate_angular_velocity_from_quat,
    compute_imu_features, process_imu_trial,
)

__all__ = [
    "normalize_quaternion", "quaternion_to_6d_rotation",
    "remove_gravity_from_acc", "calculate_angular_velocity_from_quat",
    "compute_imu_features", "process_imu_trial",
]
