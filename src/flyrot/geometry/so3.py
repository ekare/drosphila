"""Small, explicit SO(3) helpers used by FlyRot labels and metrics.

Quaternion order is always ``[x, y, z, w]``.  Rotation matrices follow the
usual active convention and are composed by matrix multiplication.
"""

from __future__ import annotations

import numpy as np


_EPS = 1e-12


def _as_float_array(value: np.ndarray | list[float]) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)


def normalize_quaternion(quaternion: np.ndarray | list[float]) -> np.ndarray:
    q = _as_float_array(quaternion)
    if q.shape[-1] != 4:
        raise ValueError(f"expected quaternion shape (..., 4), got {q.shape}")
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(norm < _EPS):
        raise ValueError("cannot normalize a zero quaternion")
    return q / norm


def quaternion_to_matrix(quaternion: np.ndarray | list[float]) -> np.ndarray:
    """Convert xyzw quaternions to rotation matrices."""

    q = normalize_quaternion(quaternion)
    x, y, z, w = np.moveaxis(q, -1, 0)
    matrix = np.empty(q.shape[:-1] + (3, 3), dtype=np.float64)
    matrix[..., 0, 0] = 1 - 2 * (y * y + z * z)
    matrix[..., 0, 1] = 2 * (x * y - z * w)
    matrix[..., 0, 2] = 2 * (x * z + y * w)
    matrix[..., 1, 0] = 2 * (x * y + z * w)
    matrix[..., 1, 1] = 1 - 2 * (x * x + z * z)
    matrix[..., 1, 2] = 2 * (y * z - x * w)
    matrix[..., 2, 0] = 2 * (x * z - y * w)
    matrix[..., 2, 1] = 2 * (y * z + x * w)
    matrix[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return matrix


def matrix_to_quaternion(matrix: np.ndarray) -> np.ndarray:
    """Convert rotation matrices to canonical xyzw quaternions."""

    matrices = _as_float_array(matrix)
    if matrices.shape[-2:] != (3, 3):
        raise ValueError(f"expected rotation matrix shape (..., 3, 3), got {matrices.shape}")
    flat = matrices.reshape(-1, 3, 3)
    result = np.empty((flat.shape[0], 4), dtype=np.float64)
    for index, r in enumerate(flat):
        trace = np.trace(r)
        if trace > 0:
            scale = 2 * np.sqrt(max(trace + 1, 0))
            result[index] = [
                (r[2, 1] - r[1, 2]) / scale,
                (r[0, 2] - r[2, 0]) / scale,
                (r[1, 0] - r[0, 1]) / scale,
                0.25 * scale,
            ]
        elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
            scale = 2 * np.sqrt(max(1 + r[0, 0] - r[1, 1] - r[2, 2], 0))
            result[index] = [
                0.25 * scale,
                (r[0, 1] + r[1, 0]) / scale,
                (r[0, 2] + r[2, 0]) / scale,
                (r[2, 1] - r[1, 2]) / scale,
            ]
        elif r[1, 1] > r[2, 2]:
            scale = 2 * np.sqrt(max(1 + r[1, 1] - r[0, 0] - r[2, 2], 0))
            result[index] = [
                (r[0, 1] + r[1, 0]) / scale,
                0.25 * scale,
                (r[1, 2] + r[2, 1]) / scale,
                (r[0, 2] - r[2, 0]) / scale,
            ]
        else:
            scale = 2 * np.sqrt(max(1 + r[2, 2] - r[0, 0] - r[1, 1], 0))
            result[index] = [
                (r[0, 2] + r[2, 0]) / scale,
                (r[1, 2] + r[2, 1]) / scale,
                0.25 * scale,
                (r[1, 0] - r[0, 1]) / scale,
            ]
    result = normalize_quaternion(result)
    result *= np.where(result[:, 3:4] < 0, -1.0, 1.0)
    return result.reshape(matrices.shape[:-2] + (4,))


def _hat(vector: np.ndarray) -> np.ndarray:
    v = _as_float_array(vector)
    if v.shape[-1] != 3:
        raise ValueError(f"expected vector shape (..., 3), got {v.shape}")
    result = np.zeros(v.shape[:-1] + (3, 3), dtype=np.float64)
    result[..., 0, 1] = -v[..., 2]
    result[..., 0, 2] = v[..., 1]
    result[..., 1, 0] = v[..., 2]
    result[..., 1, 2] = -v[..., 0]
    result[..., 2, 0] = -v[..., 1]
    result[..., 2, 1] = v[..., 0]
    return result


def exp_so3(rotation_vector: np.ndarray | list[float]) -> np.ndarray:
    """Exponential map from a rotation vector to an SO(3) matrix."""

    v = _as_float_array(rotation_vector)
    theta = np.linalg.norm(v, axis=-1)
    theta2 = theta * theta
    a = np.where(
        theta < 1e-4,
        1 - theta2 / 6 + theta2 * theta2 / 120,
        np.sin(theta) / np.where(theta == 0, 1, theta),
    )
    b = np.where(
        theta < 1e-4,
        0.5 - theta2 / 24 + theta2 * theta2 / 720,
        (1 - np.cos(theta)) / np.where(theta2 == 0, 1, theta2),
    )
    k = _hat(v)
    identity = np.broadcast_to(np.eye(3), k.shape)
    return identity + a[..., None, None] * k + b[..., None, None] * (k @ k)


def log_so3(matrix: np.ndarray) -> np.ndarray:
    """Logarithm map from an SO(3) matrix to a rotation vector."""

    quaternion = matrix_to_quaternion(matrix)
    vector = quaternion[..., :3]
    scalar = quaternion[..., 3]
    half_sine = np.linalg.norm(vector, axis=-1)
    angle = 2 * np.arctan2(half_sine, scalar)
    scale = np.where(half_sine < 1e-8, 2.0, angle / np.where(half_sine == 0, 1, half_sine))
    return vector * scale[..., None]


def relative_rotation(rotation_t: np.ndarray, rotation_next: np.ndarray) -> np.ndarray:
    """Return the next camera rotation expressed in the current camera frame."""

    return np.swapaxes(rotation_t, -1, -2) @ rotation_next


def geodesic_distance(rotation_a: np.ndarray, rotation_b: np.ndarray) -> np.ndarray:
    return np.linalg.norm(log_so3(relative_rotation(rotation_a, rotation_b)), axis=-1)


def compose_rotations(rotations: np.ndarray) -> np.ndarray:
    """Compose an ordered sequence of relative rotations from identity."""

    values = _as_float_array(rotations)
    if values.ndim < 3 or values.shape[-2:] != (3, 3):
        raise ValueError(f"expected shape (N, 3, 3), got {values.shape}")
    result = np.eye(3, dtype=np.float64)
    output = []
    for rotation in values:
        result = result @ rotation
        output.append(result.copy())
    return np.asarray(output)
