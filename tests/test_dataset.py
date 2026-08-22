import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from flyrot.data.tartanair import TartanAirWindowDataset, default_split
from flyrot.geometry.pose_conventions import load_tartanair_poses, relative_camera_rotations
from flyrot.geometry.so3 import log_so3


def test_real_rgb_pose_window_and_split_are_trajectory_safe():
    index_path = Path(os.environ.get("FLYROT_INDEX", "artifacts/tartanair2_index.json"))
    if not index_path.is_file():
        pytest.skip("set FLYROT_INDEX to run the dataset integration test")
    with index_path.open(encoding="utf-8") as handle:
        records = json.load(handle)
    splits = default_split(records)
    assert splits["train"]
    assert splits["validation"]
    assert splits["test"]
    train_trajectories = {(r["environment"], r["trajectory"]) for r in splits["train"]}
    validation_trajectories = {(r["environment"], r["trajectory"]) for r in splits["validation"]}
    assert train_trajectories.isdisjoint(validation_trajectories)
    dataset = TartanAirWindowDataset([splits["train"][0]], window_length=3, frame_gap=2, image_size=32)
    sample = dataset[0]
    assert sample["frames"].shape == (3, 3, 32, 32)
    assert sample["target_rotation_vector"].shape == (3,)
    assert sample["target_rotation_vectors"].shape == (2, 3)
    assert sample["frames"].dtype == torch.float32


def test_image_motion_target_reverses_camera_relative_direction():
    index_path = Path(os.environ.get("FLYROT_INDEX", "artifacts/tartanair2_index.json"))
    if not index_path.is_file():
        pytest.skip("set FLYROT_INDEX to run the dataset integration test")
    with index_path.open(encoding="utf-8") as handle:
        records = json.load(handle)
    record = default_split(records)["train"][0]
    camera_relative = TartanAirWindowDataset(
        [record], window_length=3, frame_gap=2, image_size=32, target_direction="camera_relative"
    )
    image_motion = TartanAirWindowDataset(
        [record], window_length=3, frame_gap=2, image_size=32, target_direction="image_motion"
    )
    sample = camera_relative[0]
    poses = load_tartanair_poses(record["pose_path"])
    expected = log_so3(relative_camera_rotations(poses[[0, 4]][::-1]))[0]
    assert torch.allclose(image_motion[0]["target_rotation_vector"], torch.from_numpy(expected).float())
    assert not torch.allclose(sample["target_rotation_vector"], image_motion[0]["target_rotation_vector"])


def test_square_input_center_crops_non_square_source_without_stretching(tmp_path):
    array = np.zeros((480, 640, 3), dtype=np.uint8)
    array[:, :60] = (255, 0, 0)
    array[:, 60:] = (0, 255, 0)
    path = tmp_path / "non_square.png"
    Image.fromarray(array).save(path)
    dataset = TartanAirWindowDataset.__new__(TartanAirWindowDataset)
    dataset.image_size = (128, 128)
    result = dataset._read_image(path)
    assert result.shape == (3, 128, 128)
    assert float(result[0].max()) == 0.0
