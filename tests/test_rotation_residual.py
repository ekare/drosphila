import torch

from flyrot.diagnostics import (
    build_rotation_residual_diagnostics,
    exact_rotational_flow,
)
from flyrot.geometry.rotational_flow import finite_difference_rotational_flow_basis


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


def test_zero_energy_is_not_high_confidence():
    energy = torch.zeros(2, 2, 24, 16, 16)
    valid = torch.ones_like(energy)
    result = build_rotation_residual_diagnostics(
        torch.zeros(2, 2, 3), energy, valid, on_energy=energy, off_energy=energy
    )
    for value in result.as_dict().values():
        assert torch.isfinite(value).all()
    assert torch.equal(result.residual_ratio, torch.ones_like(result.residual_ratio))
    assert torch.equal(result.global_confidence, torch.zeros_like(result.global_confidence))
