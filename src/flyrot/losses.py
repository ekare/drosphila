"""Rotation and uncertainty losses for FlyRot training."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def _hat(vector: torch.Tensor) -> torch.Tensor:
    result = torch.zeros(*vector.shape[:-1], 3, 3, dtype=vector.dtype, device=vector.device)
    result[..., 0, 1] = -vector[..., 2]
    result[..., 0, 2] = vector[..., 1]
    result[..., 1, 0] = vector[..., 2]
    result[..., 1, 2] = -vector[..., 0]
    result[..., 2, 0] = -vector[..., 1]
    result[..., 2, 1] = vector[..., 0]
    return result


def rotvec_to_matrix(vector: torch.Tensor) -> torch.Tensor:
    theta = torch.linalg.vector_norm(vector, dim=-1)
    theta2 = theta.square()
    safe_theta = theta.clamp_min(1e-8)
    safe_theta2 = theta2.clamp_min(1e-8)
    a = torch.where(theta < 1e-4, 1 - theta2 / 6 + theta2.square() / 120, torch.sin(theta) / safe_theta)
    b = torch.where(theta < 1e-4, 0.5 - theta2 / 24 + theta2.square() / 720, (1 - torch.cos(theta)) / safe_theta2)
    k = _hat(vector)
    identity = torch.eye(3, dtype=vector.dtype, device=vector.device).expand_as(k)
    return identity + a[..., None, None] * k + b[..., None, None] * (k @ k)


def geodesic_rotation_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_matrix = rotvec_to_matrix(prediction)
    target_matrix = rotvec_to_matrix(target)
    relative = pred_matrix.transpose(-1, -2) @ target_matrix
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1)
    sine_vector = torch.stack(
        (
            relative[..., 2, 1] - relative[..., 1, 2],
            relative[..., 0, 2] - relative[..., 2, 0],
            relative[..., 1, 0] - relative[..., 0, 1],
        ),
        dim=-1,
    ) * 0.5
    sine = torch.linalg.vector_norm(sine_vector, dim=-1)
    return torch.atan2(sine, cosine).mean()


def rotation_huber_loss(prediction: torch.Tensor, target: torch.Tensor, beta: float = 0.05) -> torch.Tensor:
    return F.smooth_l1_loss(prediction, target, beta=beta)


def heteroscedastic_nll(
    prediction: torch.Tensor,
    target: torch.Tensor,
    log_variance: torch.Tensor,
) -> torch.Tensor:
    residual = prediction - target
    return 0.5 * (torch.exp(-log_variance) * residual.square() + log_variance).mean()


def flyrot_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    log_variance: torch.Tensor,
    geodesic_weight: float = 1.0,
    huber_weight: float = 0.25,
    uncertainty_weight: float = 0.1,
) -> dict[str, torch.Tensor]:
    components = {
        "geodesic": geodesic_rotation_loss(prediction, target),
        "rotation_huber": rotation_huber_loss(prediction, target),
        "uncertainty_nll": heteroscedastic_nll(prediction, target, log_variance),
    }
    components["total"] = (
        geodesic_weight * components["geodesic"]
        + huber_weight * components["rotation_huber"]
        + uncertainty_weight * components["uncertainty_nll"]
    )
    return components
