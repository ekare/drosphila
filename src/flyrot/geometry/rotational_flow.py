"""Pinhole rotational optical-flow bases in optical camera coordinates."""

from __future__ import annotations

import torch


def _hat(vector: torch.Tensor) -> torch.Tensor:
    result = torch.zeros(*vector.shape[:-1], 3, 3, dtype=vector.dtype, device=vector.device)
    result[..., 0, 1] = -vector[..., 2]
    result[..., 0, 2] = vector[..., 1]
    result[..., 1, 0] = vector[..., 2]
    result[..., 1, 2] = -vector[..., 0]
    result[..., 2, 0] = -vector[..., 1]
    result[..., 2, 1] = vector[..., 0]
    return result


def rotational_flow_basis(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Return ``[axis, component(u,v), ...]`` for active ray rotation.

    Rays are ``[x, y, 1]`` in the right/down/forward optical frame. The
    returned fields are the first-order image displacement for a positive
    active rotation vector in that same frame.
    """

    basis_x = torch.stack((-x * y, -(1 + y.square())), dim=0)
    basis_y = torch.stack((1 + x.square(), x * y), dim=0)
    basis_z = torch.stack((-y, x), dim=0)
    return torch.stack((basis_x, basis_y, basis_z), dim=0)


def finite_difference_rotational_flow_basis(
    x: torch.Tensor, y: torch.Tensor, epsilon: float = 1e-5
) -> torch.Tensor:
    """Numerically derive the same basis from ``exp([w])`` ray rotation."""

    rays = torch.stack((x, y, torch.ones_like(x)), dim=-1)
    axes = torch.eye(3, dtype=rays.dtype, device=rays.device)
    fields = []
    for axis in axes:
        rotation = torch.matrix_exp(_hat(axis * epsilon))
        rotated = torch.einsum("ij,...j->...i", rotation, rays)
        projected = rotated[..., :2] / rotated[..., 2:].clamp_min(1e-8)
        original = rays[..., :2]
        fields.append((projected - original) / epsilon)
    return torch.stack(fields, dim=0).movedim(-1, 1)
