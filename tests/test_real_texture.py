import numpy as np
import torch

from flyrot.geometry.camera import tartanair_v2_lcam_front_intrinsics
from flyrot.real_texture import decode_direction_population, dense_homography_flow, rotation_vector, solve_weighted_rotation


def test_exact_rotation_flow_zero_and_non_square():
    intrinsics = tartanair_v2_lcam_front_intrinsics(64, 48)
    flow, valid = dense_homography_flow(np.eye(3), intrinsics)
    assert flow.shape == (48, 64, 2)
    assert np.allclose(flow, 0.0)
    assert valid.mean() == 1.0


def test_direction_decoder_preserves_polarities():
    energy = torch.zeros(1, 2, 2 * 2 * 8, 8, 8)
    energy[:, :, 0 * 16 + 0] = 1.0
    decoded = decode_direction_population(energy, (1, 2))
    assert decoded["polarity_direction"].shape == (1, 2, 2, 8, 8, 8)
    assert decoded["velocity"].shape == (1, 2, 2, 2, 8, 8)
    assert torch.isfinite(decoded["velocity"]).all()


def test_real_texture_dataset_accepts_phone_temporal_contexts():
    from flyrot.real_texture import RealTextureRotationDataset

    assert len(RealTextureRotationDataset([], window_length=9, limit=0)) == 0
    assert len(RealTextureRotationDataset([], window_length=17, limit=0)) == 0
    dataset = RealTextureRotationDataset([], image_size=(192, 108), horizontal_fov_deg=60.0, limit=0)
    assert dataset.manifest()["horizontal_fov_deg"] == 60.0


def test_weighted_rotation_solver_zero_field_is_finite():
    intrinsics = tartanair_v2_lcam_front_intrinsics(32, 24)
    field = torch.zeros(1, 1, 2, 24, 32)
    weights = torch.ones(1, 1, 24, 32)
    solved = solve_weighted_rotation(field, weights, intrinsics)
    assert torch.isfinite(solved["rotation_vector"]).all()
    assert torch.linalg.vector_norm(solved["rotation_vector"]) < 1e-5
    assert rotation_vector("yaw", 1.0, 1).shape == (3,)
