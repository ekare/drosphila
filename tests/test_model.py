import torch

from flyrot.geometry.rotational_flow import finite_difference_rotational_flow_basis, rotational_flow_basis
from flyrot.models.direction_cells import DirectionCellBank
from flyrot.models.flyrot_v0 import FlyRotV0
from flyrot.models.flyrot_v1 import FlyRotV1, compose_step_rotations
from flyrot.models.tiny_conv_baseline import TinyConvBaseline


def test_direction_cell_channel_contract_matches_forward_tensor():
    bank = DirectionCellBank(scales=(4, 8, 16))
    on = torch.rand(2, 3, 1, 16, 16)
    energy, valid, on_energy, off_energy = bank(on, on, return_components=True)
    assert bank.channels_per_polarity == 24
    assert bank.channels == energy.shape[2] == valid.shape[2] == 24
    assert bank.separate_polarity_channels == on_energy.shape[2] + off_energy.shape[2] == 48


def test_flyrot_v1_has_separate_polarity_field_and_so3_composition():
    model = FlyRotV1()
    assert sum(parameter.numel() for parameter in model.parameters()) < 50_000
    frames = torch.rand(2, 4, 3, 32, 32, requires_grad=True)
    output = model(frames)
    assert output["rotation_vector"].shape == (2, 3, 3)
    assert output["step_rotation_vector"].shape == (2, 3, 3)
    assert output["retinotopic_field"].shape == (2, 3, 2, 8, 8)
    assert output["on_motion_energy"].shape == output["off_motion_energy"].shape
    loss = output["rotation_vector"].square().mean() + output["step_rotation_vector"].square().mean() + output["log_variance"].square().mean()
    loss.backward()
    assert all(parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad)
    diagnostic = model(frames.detach(), diagnostics=True)
    assert diagnostic["directional_residual_ratio"].shape == (2, 3, 1)
    assert diagnostic["old_confidence"].shape == (2, 3, 1)


def test_compose_step_rotations_matches_matrix_product():
    steps = torch.tensor([[[0.1, 0.0, 0.0], [0.0, 0.2, 0.0]]], dtype=torch.float64)
    matrices, vectors = compose_step_rotations(steps)
    expected = torch.matrix_exp(torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, -0.1], [0.0, 0.1, 0.0]], dtype=torch.float64))
    assert torch.allclose(matrices[0, 0], expected, atol=1e-6, rtol=1e-6)
    step_y = torch.matrix_exp(torch.tensor([[0.0, 0.0, 0.2], [0.0, 0.0, 0.0], [-0.2, 0.0, 0.0]], dtype=torch.float64))
    expected_endpoint = expected @ step_y
    assert torch.allclose(matrices[0, 1], expected_endpoint, atol=5e-5, rtol=5e-5)
    assert torch.isfinite(vectors).all()


def test_rotational_flow_basis_matches_finite_difference():
    x = torch.tensor([[-0.7, 0.0, 0.6], [0.2, -0.3, 0.8]], dtype=torch.float64)
    y = torch.tensor([[-0.5, 0.4, 0.9], [0.1, 0.7, -0.2]], dtype=torch.float64)
    analytic = rotational_flow_basis(x, y)
    numeric = finite_difference_rotational_flow_basis(x, y, epsilon=1e-6)
    assert torch.allclose(analytic, numeric, atol=1e-5, rtol=1e-5)


def test_flyrot_forward_backward_and_parameter_budget():
    model = FlyRotV0()
    parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    assert parameters < 50_000
    frames = torch.rand(2, 4, 1, 32, 32, requires_grad=True)
    output = model(frames)
    assert output["rotation_vector"].shape == (2, 3, 3)
    loss = output["rotation_vector"].square().mean() + 0.01 * output["log_variance"].square().mean()
    loss.backward()
    assert all(parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad)


def test_tiny_conv_baseline_forward_backward_and_budget():
    model = TinyConvBaseline()
    assert sum(parameter.numel() for parameter in model.parameters()) < 50_000
    frames = torch.rand(2, 4, 1, 32, 32, requires_grad=True)
    output = model(frames)
    assert output["rotation_vector"].shape == (2, 3, 3)
    loss = output["rotation_vector"].square().mean() + output["log_variance"].square().mean()
    loss.backward()
    assert all(parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad)


def test_flyrot_confidence_gated_forward_backward():
    model = FlyRotV0(confidence_gated=True, readout_scale=8.0)
    frames = torch.rand(2, 4, 1, 32, 32, requires_grad=True)
    output = model(frames)
    assert output["rotation_vector"].shape == (2, 3, 3)
    assert output["confidence"].shape == (2, 3, 1)
    loss = output["rotation_vector"].square().mean() + 0.01 * output["log_variance"].square().mean()
    loss.backward()
    assert all(parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad)


def test_flyrot_magnitude_aware_forward_backward():
    model = FlyRotV0(confidence_gated=True, readout_scale=8.0, magnitude_aware=True)
    frames = torch.rand(2, 4, 1, 32, 32, requires_grad=True)
    output = model(frames)
    assert output["rotation_vector"].shape == (2, 3, 3)
    loss = output["rotation_vector"].square().mean() + 0.01 * output["log_variance"].square().mean()
    loss.backward()
    assert all(parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad)


def test_flyrot_appearance_normalized_forward_backward():
    model = FlyRotV0(confidence_gated=True, readout_scale=8.0, appearance_normalized=True)
    frames = torch.rand(2, 4, 1, 32, 32, requires_grad=True)
    output = model(frames)
    assert output["rotation_vector"].shape == (2, 3, 3)
    loss = output["rotation_vector"].square().mean() + 0.01 * output["log_variance"].square().mean()
    loss.backward()
    assert all(parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad)


def test_flyrot_scaled_forward_backward():
    model = FlyRotV0(scales=(2, 4, 8), confidence_gated=True, readout_scale=8.0)
    frames = torch.rand(2, 4, 1, 64, 64, requires_grad=True)
    output = model(frames)
    assert output["rotation_vector"].shape == (2, 3, 3)
    loss = output["rotation_vector"].square().mean() + 0.01 * output["log_variance"].square().mean()
    loss.backward()
    assert all(parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad)


def test_flyrot_scale_separated_magnitude_forward_backward():
    model = FlyRotV0(
        scales=(4, 8, 16),
        confidence_gated=True,
        readout_scale=8.0,
        magnitude_aware=True,
        scale_separated=True,
    )
    frames = torch.rand(2, 4, 1, 64, 64, requires_grad=True)
    output = model(frames)
    assert output["rotation_vector"].shape == (2, 3, 3)
    loss = output["rotation_vector"].square().mean() + 0.01 * output["log_variance"].square().mean()
    loss.backward()
    assert all(parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad)


def test_flyrot_least_squares_basis_forward_backward():
    model = FlyRotV0(scales=(4, 8, 16), confidence_gated=True, readout_scale=8.0, least_squares_basis=True)
    frames = torch.rand(2, 4, 1, 32, 32, requires_grad=True)
    output = model(frames)
    assert output["rotation_vector"].shape == (2, 3, 3)
    (output["rotation_vector"].square().mean() + 0.01 * output["log_variance"].square().mean()).backward()
    assert all(parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad)


def test_flyrot_motion_consistency_gate_forward_backward():
    model = FlyRotV0(scales=(4, 8, 16), readout_scale=1.0, motion_consistency_gate=True)
    frames = torch.rand(2, 4, 1, 32, 32, requires_grad=True)
    output = model(frames)
    assert output["rotation_vector"].shape == (2, 3, 3)
    assert output["motion_coherence"].shape == (2, 3, 1)
    (output["rotation_vector"].square().mean() + output["motion_coherence"].mean() + 0.01 * output["log_variance"].square().mean()).backward()
    assert all(parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad)


def test_flyrot_input_dependent_uncertainty_forward_backward():
    model = FlyRotV0(scales=(4, 8, 16), readout_scale=1.0, input_dependent_uncertainty=True)
    frames = torch.rand(2, 4, 1, 32, 32, requires_grad=True)
    output = model(frames)
    assert output["log_variance"].shape == (2, 3, 3)
    loss = output["rotation_vector"].square().mean() + output["log_variance"].square().mean()
    loss.backward()
    assert all(parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad)


def test_flyrot_magnitude_confidence_is_bounded_and_input_dependent():
    model = FlyRotV0(scales=(4, 8, 16), readout_scale=1.0, magnitude_confidence=True)
    output = model(torch.rand(4, 4, 1, 32, 32))
    confidence = output["confidence"]
    assert torch.isfinite(confidence).all()
    assert float(confidence.min()) >= 0.0
    assert float(confidence.max()) <= 1.0
    assert float(confidence.std()) > 0.0


def test_flyrot_intrinsics_stretched_forward_backward():
    model = FlyRotV0(confidence_gated=True, readout_scale=8.0, focal_y_over_x=4.0 / 3.0)
    frames = torch.rand(2, 4, 1, 64, 64, requires_grad=True)
    output = model(frames)
    assert output["rotation_vector"].shape == (2, 3, 3)
    loss = output["rotation_vector"].square().mean() + 0.01 * output["log_variance"].square().mean()
    loss.backward()
    assert all(parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad)


def test_flyrot_diagnostics_flag_preserves_default_path_and_exposes_components():
    model = FlyRotV0(scales=(4, 8, 16), magnitude_confidence=True)
    frames = torch.rand(2, 4, 1, 32, 32)
    default = model(frames)
    diagnostic = model(frames, diagnostics=True)

    assert diagnostic["rotation_vector"].shape == default["rotation_vector"].shape
    assert torch.allclose(diagnostic["rotation_vector"], default["rotation_vector"])
    assert diagnostic["on_motion_energy"].shape == diagnostic["direction_energy"].shape
    assert diagnostic["off_motion_energy"].shape == diagnostic["direction_energy"].shape
    assert diagnostic["residual_ratio"].shape == (2, 3, 1)
    assert diagnostic["axis_confidence"].shape == (2, 3, 3)
    assert diagnostic["global_confidence"].shape == (2, 3, 1)
    for key in (
        "on_motion_energy",
        "off_motion_energy",
        "residual_ratio",
        "axis_confidence",
        "global_confidence",
    ):
        assert torch.isfinite(diagnostic[key]).all()
