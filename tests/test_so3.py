import numpy as np

from flyrot.geometry.so3 import (
    compose_rotations,
    exp_so3,
    geodesic_distance,
    log_so3,
    matrix_to_quaternion,
    quaternion_to_matrix,
)


def test_quaternion_roundtrip_and_normalization():
    quaternion = np.array([0.2, -0.3, 0.4, 0.5])
    matrix = quaternion_to_matrix(quaternion)
    recovered = matrix_to_quaternion(matrix)
    assert np.allclose(quaternion_to_matrix(recovered), matrix, atol=1e-10)
    assert np.isclose(np.linalg.norm(recovered), 1.0)


def test_known_z_rotation_sign():
    vector = np.array([0.0, 0.0, np.pi / 2])
    matrix = exp_so3(vector)
    assert np.allclose(matrix @ np.array([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-10)
    assert np.allclose(log_so3(matrix), vector, atol=1e-10)


def test_geodesic_and_composition():
    first = exp_so3([0.1, 0.0, 0.0])
    second = exp_so3([0.0, 0.2, 0.0])
    composed = compose_rotations(np.stack([first, second]))
    assert composed.shape == (2, 3, 3)
    assert np.isclose(geodesic_distance(np.eye(3), composed[0]), 0.1, atol=1e-10)
    assert np.isclose(geodesic_distance(composed[0], composed[1]), 0.2, atol=1e-10)
