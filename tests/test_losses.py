import torch

from flyrot.losses import flyrot_loss, geodesic_rotation_loss, rotvec_to_matrix
from flyrot.metrics import rotation_metrics, selective_risk


def test_torch_so3_and_loss_are_finite_and_differentiable():
    prediction = torch.tensor([[0.1, -0.05, 0.02]], requires_grad=True)
    target = torch.tensor([[0.0, 0.0, 0.0]])
    log_variance = torch.zeros_like(prediction, requires_grad=True)
    matrix = rotvec_to_matrix(prediction)
    assert matrix.shape == (1, 3, 3)
    components = flyrot_loss(prediction, target, log_variance)
    assert torch.isfinite(components["total"])
    assert geodesic_rotation_loss(target, target) < 1e-6
    components["total"].backward()
    assert prediction.grad is not None
    assert log_variance.grad is not None


def test_rotation_metrics_and_selective_risk():
    target = torch.tensor([[0.1, 0.0, 0.0], [0.0, 0.2, 0.0], [0.0, 0.0, 0.0]])
    prediction = torch.tensor([[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.01, 0.0, 0.0]])
    confidence = torch.tensor([0.9, 0.8, 0.1])
    metrics = rotation_metrics(prediction, target, confidence)
    assert metrics["samples"] == 3
    assert metrics["selective_risk"][0]["samples"] == 1
    assert len(selective_risk(torch.ones(3), confidence)) == 4
