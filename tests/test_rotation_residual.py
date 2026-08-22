import torch

from flyrot.diagnostics import (
    build_rotation_residual_diagnostics,
    estimate_rotation_evidence,
    exact_rotational_flow,
)
from flyrot.geometry.camera import CropResizeTransform, tartanair_v2_lcam_front_intrinsics
from flyrot.geometry.rotational_flow import CameraIntrinsics, finite_difference_rotational_flow_basis
from flyrot.geometry.translation_flow import rigid_camera_flow, scale_free_translation_direction


def _normalized_grid(height: int, width: int) -> tuple[torch.Tensor, torch.Tensor]:
    y = (torch.arange(height, dtype=torch.float32) - (height - 1) / 2) / (height / 2)
    x = (torch.arange(width, dtype=torch.float32) - (width - 1) / 2) / (width / 2)
    return torch.meshgrid(y, x, indexing="ij")


def _directional_energy(rotation: torch.Tensor, height: int, width: int) -> torch.Tensor:
    flow = exact_rotational_flow(rotation, height, width)[..., :]
    unit = flow / torch.linalg.vector_norm(flow, dim=2, keepdim=True).clamp_min(1e-8)
    angles = torch.arange(8, dtype=flow.dtype) * (2 * torch.pi / 8)
    directions = torch.stack((angles.cos(), angles.sin()), dim=1)
    compatibility = torch.einsum("dp,btphw->btdhw", directions, unit).clamp_min(0)
    return compatibility.repeat(1, 1, 3, 1, 1)


def test_exact_rotation_flow_matches_finite_difference_basis():
    height, width = 7, 9
    y, x = _normalized_grid(height, width)
    basis = finite_difference_rotational_flow_basis(x, y, epsilon=1e-5)
    for axis in range(3):
        rotation = torch.zeros(1, 3)
        rotation[0, axis] = 1e-5
        predicted = exact_rotational_flow(rotation, height, width)[0] / 1e-5
        assert torch.allclose(predicted, basis[axis], atol=2e-4, rtol=2e-4)


def test_exact_rotation_flow_sign_and_zero_case():
    zero = exact_rotational_flow(torch.zeros(2, 3), 9, 9)
    assert torch.equal(zero, torch.zeros_like(zero))
    rotation = torch.tensor([[0.0, 0.0, 1e-3]])
    flow = exact_rotational_flow(rotation, 9, 9)[0]
    assert flow[0, 0, 0] > 0  # positive rz moves the upper-left ray rightward
    assert flow[1, 0, 0] < 0  # and upward in the right/down image convention


def test_full_intrinsics_centered_legacy_equivalence_and_crop_resize():
    height, width = 11, 13
    rotation = torch.tensor([[0.03, -0.02, 0.01]])
    legacy = exact_rotational_flow(rotation, height, width, focal_y_over_x=1.25)
    intrinsics = CameraIntrinsics.centered(width, height, focal_y_over_x=1.25)
    full = exact_rotational_flow(rotation, intrinsics=intrinsics)
    assert torch.allclose(legacy, full, atol=1e-6, rtol=1e-6)
    shifted = intrinsics.cropped(2, 1, width - 2, height - 1)
    assert shifted.cx == intrinsics.cx - 2
    assert shifted.cy == intrinsics.cy - 1
    resized = intrinsics.resized(26, 22)
    assert resized.width == 26 and resized.height == 22
    assert torch.equal(exact_rotational_flow(torch.zeros(1, 3), intrinsics=shifted), torch.zeros(1, 2, 10, 11))


def test_exact_homography_golden_matches_non_square_off_center_intrinsics():
    intrinsics = CameraIntrinsics(fx=17.0, fy=13.0, cx=4.25, cy=3.75, width=11, height=9)
    rotation = torch.tensor([[0.09, -0.05, 0.04]], dtype=torch.float64)
    flow = exact_rotational_flow(rotation, intrinsics=intrinsics)[0]
    u = torch.arange(intrinsics.width, dtype=torch.float64)
    v = torch.arange(intrinsics.height, dtype=torch.float64)
    vv, uu = torch.meshgrid(v, u, indexing="ij")
    rays = torch.stack(
        ((uu - intrinsics.cx) / intrinsics.fx, (vv - intrinsics.cy) / intrinsics.fy, torch.ones_like(uu)), dim=-1
    )
    hat = torch.zeros(3, 3, dtype=torch.float64)
    hat[0, 1], hat[0, 2], hat[1, 0], hat[1, 2], hat[2, 0], hat[2, 1] = (
        -rotation[0, 2], rotation[0, 1], rotation[0, 2], -rotation[0, 0], -rotation[0, 1], rotation[0, 0]
    )
    rotated = torch.einsum("ij,hwj->hwi", torch.matrix_exp(hat), rays)
    projected_u = rotated[..., 0] / rotated[..., 2] * intrinsics.fx + intrinsics.cx
    projected_v = rotated[..., 1] / rotated[..., 2] * intrinsics.fy + intrinsics.cy
    expected = torch.stack((projected_u - uu, projected_v - vv), dim=0)
    pixel_scale = torch.tensor([intrinsics.fx, intrinsics.fy], dtype=torch.float64).view(2, 1, 1)
    assert torch.allclose(flow * pixel_scale, expected, atol=1e-10, rtol=1e-10)
    wrong_sign = exact_rotational_flow(-rotation, intrinsics=intrinsics)[0]
    assert torch.median(torch.sum(flow * expected, dim=0)) > 0
    assert torch.median(torch.sum(wrong_sign * expected, dim=0)) < 0


def test_crop_resize_transform_preserves_ray_geometry():
    source = tartanair_v2_lcam_front_intrinsics()
    transform = CropResizeTransform(left=80, top=40, crop_width=400, crop_height=320, output_width=200, output_height=160)
    output = transform.apply(source)
    assert output.width == 200 and output.height == 160
    assert output.fx == 160.0 and output.fy == 160.0
    assert output.cx == 119.75 and output.cy == 139.75


def test_depth_rigid_flow_decomposes_rotation_and_translation():
    intrinsics = CameraIntrinsics(fx=16.0, fy=14.0, cx=5.0, cy=4.0, width=10, height=8)
    depth = torch.full((1, 8, 10), 4.0, dtype=torch.float64)
    rotation = torch.tensor([[0.03, -0.02, 0.01]], dtype=torch.float64)
    translation = torch.tensor([[0.2, -0.1, 0.05]], dtype=torch.float64)
    result = rigid_camera_flow(rotation, translation, depth, intrinsics)
    pure = rigid_camera_flow(rotation, torch.zeros_like(translation), depth, intrinsics)
    assert result["valid"].all()
    assert torch.allclose(pure["full_flow"], pure["rotation_flow"], atol=1e-10, rtol=1e-10)
    assert torch.allclose(result["full_flow"], result["rotation_flow"] + result["translation_flow"], atol=1e-10, rtol=1e-10)
    direction = scale_free_translation_direction(translation)
    assert torch.allclose(torch.linalg.vector_norm(direction, dim=-1), torch.ones(1, dtype=torch.float64))


def test_native_energy_residual_is_small_for_matching_rotation_and_large_for_wrong_rotation():
    height, width = 32, 32
    rotation = torch.tensor([[0.12, -0.08, 0.05]])
    energy = _directional_energy(rotation[:, None], height, width)
    valid = torch.ones_like(energy)
    on = energy.clone()
    off = energy.clone()
    matching = build_rotation_residual_diagnostics(rotation[:, None], energy, valid, on_energy=on, off_energy=off)
    wrong = build_rotation_residual_diagnostics(-rotation[:, None], energy, valid, on_energy=on, off_energy=off)
    assert torch.isfinite(matching.residual_ratio).all()
    assert matching.residual_ratio.item() < 0.25
    assert wrong.residual_ratio.item() > matching.residual_ratio.item() + 0.25
    assert matching.explained_total_energy.sum() > wrong.explained_total_energy.sum()
    assert torch.allclose(matching.directional_residual_ratio, matching.residual_ratio, equal_nan=True)
    assert torch.isfinite(matching.directional_fit_score).all()


def test_rotation_evidence_has_per_scale_geometry_and_validity():
    height, width = 24, 24
    rotation = torch.tensor([[0.12, -0.08, 0.05]])
    energy = _directional_energy(rotation[:, None], height, width)
    result = estimate_rotation_evidence(energy, torch.ones_like(energy))
    assert result["rotation_evidence"].shape == (1, 1, 3, 3)
    assert result["information_matrix"].shape == (1, 1, 3, 3, 3)
    assert result["evidence_valid"].all()
    assert result["aggregate_valid"].all()


def test_temporal_validity_requires_two_real_motion_states():
    height, width = 24, 24
    rotation = torch.tensor([0.08, -0.06, 0.05])
    rotations = torch.stack((torch.zeros(3), rotation, rotation, rotation)).view(1, 4, 3)
    energy = _directional_energy(rotations, height, width)
    result = build_rotation_residual_diagnostics(rotations, energy, torch.ones_like(energy), on_energy=energy, off_energy=energy)
    assert not result.temporal_agreement_valid[:, 0].any()
    assert result.valid_motion_state_count.item() == 3
    assert result.valid_temporal_pair_count.item() == 2
    assert result.temporal_agreement_valid[:, 2:].all()
    assert torch.nanmean(result.temporal_agreement_score[:, 2:]) > 0.9

    short_rotations = rotations[:, :2]
    short_energy = energy[:, :2]
    short = build_rotation_residual_diagnostics(
        short_rotations, short_energy, torch.ones_like(short_energy), on_energy=short_energy, off_energy=short_energy
    )
    assert not short.temporal_agreement_valid.any()
    assert torch.isnan(short.temporal_agreement_score).all()


def test_on_off_and_scale_validity_are_geometry_based():
    height, width = 24, 24
    rotation = torch.tensor([[0.08, -0.06, 0.05]])
    energy = _directional_energy(rotation[:, None], height, width)
    valid = torch.ones_like(energy)
    shifted_on = torch.roll(energy, shifts=(1, -1), dims=(-2, -1))
    result = build_rotation_residual_diagnostics(
        rotation[:, None], energy, valid, on_energy=shifted_on, off_energy=energy
    )
    assert result.on_off_agreement_valid.all()
    assert result.on_off_agreement_score[0, 0, 0] > 0.8
    one_channel = build_rotation_residual_diagnostics(
        rotation[:, None], energy, valid, on_energy=energy, off_energy=torch.zeros_like(energy)
    )
    assert not one_channel.on_off_agreement_valid.any()
    assert one_channel.on_off_balance_score.max() < 1e-4
    assert torch.isfinite(one_channel.global_reliability).all()

    one_scale = energy.clone()
    one_scale[:, :, 8:] = 0
    one_scale_result = build_rotation_residual_diagnostics(
        rotation[:, None], one_scale, valid, on_energy=one_scale, off_energy=one_scale
    )
    assert (one_scale_result.active_scale_count == 1).all()
    assert not one_scale_result.scale_agreement_valid.any()
    assert torch.isnan(one_scale_result.scale_agreement_score).all()


def test_observability_axis_fields_are_not_sorted_eigenvalue_labels():
    height, width = 20, 28
    rotation = torch.tensor([[0.08, -0.06, 0.05]])
    energy = _directional_energy(rotation[:, None], height, width)
    energy[..., :, : width // 3] = 0
    result = build_rotation_residual_diagnostics(rotation[:, None], energy, torch.ones_like(energy), on_energy=energy, off_energy=energy)
    assert torch.all(result.observability_eigenvalues[..., 1:] >= result.observability_eigenvalues[..., :-1])
    assert torch.allclose(result.axis_confidence.sum(dim=-1), torch.ones_like(result.axis_confidence[..., 0]))
    assert result.axis_variance.shape[-1] == 3
    assert result.axis_information.shape[-1] == 3


def test_zero_energy_is_not_high_confidence():
    energy = torch.zeros(2, 2, 24, 16, 16)
    valid = torch.ones_like(energy)
    result = build_rotation_residual_diagnostics(
        torch.zeros(2, 2, 3), energy, valid, on_energy=energy, off_energy=energy
    )
    intentional_unavailable = {
        "scale_agreement_score",
        "scale_agreement",
        "temporal_agreement_score",
        "temporal_agreement",
        "on_off_agreement_score",
        "on_off_agreement",
        "observability_score",
    }
    for key, value in result.as_dict().items():
        if key in intentional_unavailable:
            assert torch.isnan(value).all()
        else:
            assert torch.isfinite(value).all()
    assert torch.equal(result.residual_ratio, torch.ones_like(result.residual_ratio))
    assert torch.equal(result.global_reliability, torch.zeros_like(result.global_reliability))
    assert not result.global_reliability_valid.any()
