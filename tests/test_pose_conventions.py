import os
from pathlib import Path

import numpy as np
import pytest

from flyrot.geometry.pose_conventions import (
    load_tartanair_poses,
    pose_rows_to_world_camera,
    relative_camera_rotation_vectors,
)
from flyrot.geometry.so3 import exp_so3


def test_tartanair_rows_use_xyzw_and_relative_camera_rotation():
    rows = np.array(
        [
            [0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, np.sin(np.pi / 4), np.cos(np.pi / 4)],
        ],
        dtype=np.float64,
    )
    relative = relative_camera_rotation_vectors(rows)
    assert np.allclose(relative[0], [0, 0, np.pi / 2], atol=1e-10)


def test_pose_matrix_translation_and_rotation():
    rows = np.array([[1, 2, 3, 0, 0, 0, 1]], dtype=np.float64)
    matrix = pose_rows_to_world_camera(rows)[0]
    assert np.allclose(matrix[:3, :3], np.eye(3))
    assert np.allclose(matrix[:3, 3], [1, 2, 3])


def test_real_tartanair_pose_file_has_matching_shape():
    path = Path(os.environ.get("FLYROT_POSE_FILE", "data/tartanair-v2/House/Data_easy/P000/pose_lcam_front.txt"))
    if not path.is_file():
        pytest.skip("set FLYROT_POSE_FILE to run the dataset pose integration test")
    poses = load_tartanair_poses(path)
    assert poses.shape == (680, 7)
    assert np.allclose(np.linalg.norm(poses[:, 3:7], axis=1), 1.0, atol=1e-5)
