"""TartanAir pose parsing and explicit camera-relative target conventions."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .so3 import quaternion_to_matrix, relative_rotation


TARTANAIR_POSE_FORMAT = "tx ty tz qx qy qz qw"
FRAME_DESCRIPTION = "NED: x forward, y right, z down"


def load_tartanair_poses(path: str | Path) -> np.ndarray:
    poses = np.loadtxt(path, dtype=np.float64)
    if poses.ndim == 1:
        poses = poses[None, :]
    if poses.ndim != 2 or poses.shape[1] != 7:
        raise ValueError(f"expected TartanAir pose array (N, 7), got {poses.shape}")
    if not np.isfinite(poses).all():
        raise ValueError(f"non-finite pose value in {path}")
    quaternion_norms = np.linalg.norm(poses[:, 3:7], axis=1)
    if np.any(quaternion_norms < 1e-10):
        raise ValueError(f"zero quaternion in {path}")
    return poses


def pose_rows_to_world_camera(poses: np.ndarray) -> np.ndarray:
    """Convert tx ty tz qx qy qz qw rows to ``T_world_camera`` matrices."""

    rows = np.asarray(poses, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1] != 7:
        raise ValueError(f"expected pose rows (N, 7), got {rows.shape}")
    result = np.broadcast_to(np.eye(4), (len(rows), 4, 4)).copy()
    result[:, :3, :3] = quaternion_to_matrix(rows[:, 3:7])
    result[:, :3, 3] = rows[:, :3]
    return result


def relative_camera_rotations(poses: np.ndarray) -> np.ndarray:
    """Compute ``R_t.T @ R_t+1`` for consecutive TartanAir poses."""

    matrices = pose_rows_to_world_camera(poses)
    return relative_rotation(matrices[:-1, :3, :3], matrices[1:, :3, :3])


def relative_camera_rotation_vectors(poses: np.ndarray) -> np.ndarray:
    from .so3 import log_so3

    return log_so3(relative_camera_rotations(poses))
