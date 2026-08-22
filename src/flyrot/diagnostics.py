"""Rotation-residual diagnostics in the model's native motion-energy space.

The direction-cell output is not optical flow.  It is non-negative local
direction evidence, so this module keeps the scientific residual in that
space and exposes pseudo-flow-like vectors only as explicitly diagnostic
visualizations.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from flyrot.geometry.rotational_flow import _hat, rotational_flow_basis


@dataclass
class RotationResidualDiagnostics:
    """Structured outputs for one batch of causal model states."""

    predicted_rotation: torch.Tensor
    predicted_rotation_flow: torch.Tensor
    on_motion_energy: torch.Tensor
    off_motion_energy: torch.Tensor
    observed_motion_energy: torch.Tensor
    explained_rotation_energy: torch.Tensor
    residual_motion_energy: torch.Tensor
    observed_total_energy: torch.Tensor
    explained_total_energy: torch.Tensor
    residual_total_energy: torch.Tensor
    residual_ratio: torch.Tensor
    observed_pseudo_flow: torch.Tensor
    rotation_pseudo_flow: torch.Tensor
    residual_pseudo_flow: torch.Tensor
    residual_energy_by_scale: torch.Tensor
    spatial_support: torch.Tensor
    scale_agreement: torch.Tensor
    temporal_agreement: torch.Tensor
    on_off_agreement: torch.Tensor
    observability_matrix: torch.Tensor
    observability_eigenvalues: torch.Tensor
    observability_condition: torch.Tensor
    axis_confidence: torch.Tensor
    global_confidence: torch.Tensor
    motion_presence: torch.Tensor

    def as_dict(self) -> dict[str, torch.Tensor]:
        """Return tensor fields using stable, descriptive output names."""

        return {
            "predicted_rotation": self.predicted_rotation,
            "predicted_rotation_flow": self.predicted_rotation_flow,
            "on_motion_energy": self.on_motion_energy,
            "off_motion_energy": self.off_motion_energy,
            "observed_motion_energy": self.observed_motion_energy,
            "explained_rotation_energy": self.explained_rotation_energy,
            "residual_motion_energy": self.residual_motion_energy,
            "observed_total_energy": self.observed_total_energy,
            "explained_total_energy": self.explained_total_energy,
            "residual_total_energy": self.residual_total_energy,
            "residual_ratio": self.residual_ratio,
            "observed_pseudo_flow": self.observed_pseudo_flow,
            "rotation_pseudo_flow": self.rotation_pseudo_flow,
            "residual_pseudo_flow": self.residual_pseudo_flow,
            "residual_energy_by_scale": self.residual_energy_by_scale,
            "spatial_support": self.spatial_support,
            "scale_agreement": self.scale_agreement,
            "temporal_agreement": self.temporal_agreement,
            "on_off_agreement": self.on_off_agreement,
            "observability_matrix": self.observability_matrix,
            "observability_eigenvalues": self.observability_eigenvalues,
            "observability_condition": self.observability_condition,
            "axis_confidence": self.axis_confidence,
            "global_confidence": self.global_confidence,
            "motion_presence": self.motion_presence,
        }


def _direction_vectors(
    directions: int, *, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    angles = torch.arange(directions, device=device, dtype=dtype) * (2.0 * torch.pi / directions)
    return torch.stack((angles.cos(), angles.sin()), dim=1)


def _safe_depth(depth: torch.Tensor, minimum: float = 1e-6) -> torch.Tensor:
    sign = torch.where(depth < 0, -torch.ones_like(depth), torch.ones_like(depth))
    return torch.where(depth.abs() < minimum, sign * minimum, depth)


def exact_rotational_flow(
    rotation_vector: torch.Tensor,
    height: int,
    width: int,
    focal_y_over_x: float = 1.0,
) -> torch.Tensor:
    """Project normalized camera rays after an active SO(3) rotation.

    ``rotation_vector`` may have shape ``(..., 3)``.  The result has shape
    ``(..., 2, H, W)`` and is a normalized-coordinate displacement, not a
    pixel/second velocity.  The active-ray convention matches the rotational
    basis used by :class:`flyrot.models.rotation_evidence.RotationEvidence`.
    """

    if rotation_vector.ndim < 1 or rotation_vector.shape[-1] != 3:
        raise ValueError(f"expected (..., 3) rotation vectors, got {tuple(rotation_vector.shape)}")
    if height < 1 or width < 1 or focal_y_over_x <= 0:
        raise ValueError("height, width, and focal_y_over_x must be positive")
    device, dtype = rotation_vector.device, rotation_vector.dtype
    y_pixels = torch.arange(height, device=device, dtype=dtype) - (height - 1) / 2.0
    x_pixels = torch.arange(width, device=device, dtype=dtype) - (width - 1) / 2.0
    yy, xx = torch.meshgrid(
        y_pixels / ((height / 2.0) * focal_y_over_x),
        x_pixels / (width / 2.0),
        indexing="ij",
    )
    rays = torch.stack((xx, yy, torch.ones_like(xx)), dim=-1)
    rotation = torch.matrix_exp(_hat(rotation_vector))
    rotated = torch.einsum("...ij,hwj->...hwi", rotation, rays)
    projected = rotated[..., :2] / _safe_depth(rotated[..., 2:3])
    flow = projected - rays[..., :2]
    return flow.movedim(-1, -3)


def _cosine(a: torch.Tensor, b: torch.Tensor, dim: int = -1) -> torch.Tensor:
    numerator = (a * b).sum(dim=dim)
    denominator = torch.linalg.vector_norm(a, dim=dim) * torch.linalg.vector_norm(b, dim=dim)
    return numerator / denominator.clamp_min(1e-8)


def _spatial_entropy_support(total_energy: torch.Tensor) -> torch.Tensor:
    """Return normalized effective spatial support in ``B,T,1`` format."""

    b, t, h, w = total_energy.shape
    flat = total_energy.reshape(b, t, h * w).clamp_min(0)
    total = flat.sum(dim=-1, keepdim=True)
    probabilities = flat / total.clamp_min(1e-8)
    entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(dim=-1)
    support = entropy / torch.log(torch.tensor(float(h * w), device=flat.device, dtype=flat.dtype)).clamp_min(1.0)
    return torch.where(total.squeeze(-1) > 1e-8, support, torch.zeros_like(support)).unsqueeze(-1)


def _pairwise_direction_agreement(vectors: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Weighted agreement of ``B,T,S,2,H,W`` vectors across scale."""

    b, t, scales, _, h, w = vectors.shape
    weighted_mean = (vectors * weights.unsqueeze(3)).sum(dim=2)
    mean_norm = torch.linalg.vector_norm(weighted_mean, dim=2, keepdim=True)
    mean_unit = weighted_mean / mean_norm.clamp_min(1e-8)
    unit = vectors / torch.linalg.vector_norm(vectors, dim=3, keepdim=True).clamp_min(1e-8)
    cosine = (unit * mean_unit.unsqueeze(2)).sum(dim=3)
    weighted = (cosine.clamp_min(0) * weights).sum(dim=(2, 3, 4))
    denominator = weights.sum(dim=(2, 3, 4)).clamp_min(1e-8)
    return (weighted / denominator).unsqueeze(-1)


def build_rotation_residual_diagnostics(
    rotation_vector: torch.Tensor,
    energy: torch.Tensor,
    valid: torch.Tensor,
    *,
    on_energy: torch.Tensor | None = None,
    off_energy: torch.Tensor | None = None,
    focal_y_over_x: float = 1.0,
    energy_floor: float = 1e-3,
) -> RotationResidualDiagnostics:
    """Decompose direction energy into rotation-compatible and residual parts.

    The decomposition uses positive compatibility between each observed
    direction cell and the exact perspective flow direction generated by the
    predicted rotation. It deliberately does not claim that energy magnitude
    is a pixel displacement magnitude.
    """

    if energy.ndim != 5 or valid.shape != energy.shape:
        raise ValueError("energy and valid must both have shape B,T,C,H,W")
    if rotation_vector.ndim != 3 or rotation_vector.shape[:2] != energy.shape[:2]:
        raise ValueError("rotation_vector must have shape B,T,3 matching energy")
    b, t, channels, height, width = energy.shape
    if channels % 8 != 0:
        raise ValueError("direction energy channel count must be a multiple of 8")
    directions = 8
    scales = channels // directions
    if energy_floor <= 0:
        raise ValueError("energy_floor must be positive")
    observed = energy.clamp_min(0) * valid.to(dtype=energy.dtype).clamp_min(0)
    observed_by_scale = observed.reshape(b, t, scales, directions, height, width)
    total_by_pixel = observed_by_scale.sum(dim=(2, 3))
    observed_total = total_by_pixel.unsqueeze(2)
    observed_scalar = total_by_pixel.sum(dim=(2, 3), keepdim=False).unsqueeze(-1)

    predicted_flow = exact_rotational_flow(rotation_vector, height, width, focal_y_over_x)
    flow_unit = predicted_flow / torch.linalg.vector_norm(predicted_flow, dim=2, keepdim=True).clamp_min(1e-8)
    vectors = _direction_vectors(directions, device=energy.device, dtype=energy.dtype)
    compatibility = torch.einsum("dp,btphw->btdhw", vectors, flow_unit).clamp_min(0)
    compatibility = compatibility * (torch.linalg.vector_norm(predicted_flow, dim=2, keepdim=True) > 1e-8)
    compatibility = compatibility.unsqueeze(2).expand(b, t, scales, directions, height, width)
    explained = observed_by_scale * compatibility
    residual = (observed_by_scale - explained).clamp_min(0)
    explained_total = explained.sum(dim=(2, 3)).unsqueeze(2)
    residual_total = residual.sum(dim=(2, 3)).unsqueeze(2)
    residual_scalar = residual_total.sum(dim=(3, 4), keepdim=False)
    residual_ratio = residual_scalar / observed_scalar.clamp_min(1e-8)
    residual_ratio = torch.where(observed_scalar > 1e-8, residual_ratio, torch.ones_like(residual_ratio)).clamp(0, 1)

    observed_vector = torch.einsum("btsdhw,dp->btphw", observed_by_scale, vectors)
    expected_vector = torch.einsum("btsdhw,dp->btphw", compatibility * observed_by_scale, vectors)
    residual_vector = torch.einsum("btsdhw,dp->btphw", residual, vectors)

    scale_vectors = torch.einsum("btsdhw,dp->btsphw", observed_by_scale, vectors)
    scale_weights = observed_by_scale.sum(dim=3)
    scale_agreement = _pairwise_direction_agreement(scale_vectors, scale_weights)

    temporal_vector = observed_vector
    if t == 1:
        temporal_agreement = torch.ones((b, 1, 1), device=energy.device, dtype=energy.dtype)
    else:
        temporal_cosine = _cosine(temporal_vector[:, 1:], temporal_vector[:, :-1], dim=2)
        temporal_weights = total_by_pixel[:, 1:]
        pair_agreement = (temporal_cosine.clamp_min(0) * temporal_weights).sum(dim=(1, 2, 3)) / temporal_weights.sum(dim=(1, 2, 3)).clamp_min(1e-8)
        temporal_agreement = torch.cat(
            (
                torch.ones((b, 1, 1), device=energy.device, dtype=energy.dtype),
                pair_agreement.view(b, 1, 1).expand(b, t - 1, 1),
            ),
            dim=1,
        )

    if on_energy is None or off_energy is None:
        on_off_agreement = torch.ones((b, t, 1), device=energy.device, dtype=energy.dtype)
    else:
        if on_energy.shape != energy.shape or off_energy.shape != energy.shape:
            raise ValueError("on_energy and off_energy must match energy shape")
        on = on_energy.clamp_min(0).reshape(b, t, scales, directions, height, width)
        off = off_energy.clamp_min(0).reshape(b, t, scales, directions, height, width)
        on_vector = torch.einsum("btsdhw,dp->btsphw", on, vectors).sum(dim=2)
        off_vector = torch.einsum("btsdhw,dp->btsphw", off, vectors).sum(dim=2)
        union = (on.sum(dim=(2, 3)) + off.sum(dim=(2, 3)))
        agreement = _cosine(on_vector, off_vector, dim=2).clamp_min(0)
        on_off_agreement = (agreement * union).sum(dim=(2, 3)) / union.sum(dim=(2, 3)).clamp_min(1e-8)
        on_off_agreement = torch.where(union.sum(dim=(2, 3)) > 1e-8, on_off_agreement, torch.zeros_like(on_off_agreement)).unsqueeze(-1)

    y_pixels = torch.arange(height, device=energy.device, dtype=energy.dtype) - (height - 1) / 2.0
    x_pixels = torch.arange(width, device=energy.device, dtype=energy.dtype) - (width - 1) / 2.0
    yy, xx = torch.meshgrid(
        y_pixels / ((height / 2.0) * focal_y_over_x),
        x_pixels / (width / 2.0),
        indexing="ij",
    )
    basis = rotational_flow_basis(xx, yy).permute(2, 3, 1, 0).contiguous()
    # basis is H,W,2,3; the normalized matrix is the weighted J^T J.
    observability = torch.einsum("bthw,hwij,hwik->btjk", total_by_pixel, basis, basis)
    total_scalar = total_by_pixel.sum(dim=(2, 3), keepdim=False).unsqueeze(-1)
    observability = observability / total_scalar.unsqueeze(-1).clamp_min(1e-8)
    regularizer = torch.eye(3, device=energy.device, dtype=energy.dtype).view(1, 1, 3, 3) * 1e-6
    eigenvalues = torch.linalg.eigvalsh(observability + regularizer).clamp_min(0)
    condition = eigenvalues[..., -1] / eigenvalues[..., 0].clamp_min(1e-8)
    axis_confidence = eigenvalues / eigenvalues.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    axis_confidence = torch.where(total_scalar > 1e-8, axis_confidence, torch.zeros_like(axis_confidence))
    observability_global = eigenvalues[..., 0] / eigenvalues[..., -1].clamp_min(1e-8)
    observability_global = observability_global.unsqueeze(-1).clamp(0, 1)

    support = _spatial_entropy_support(total_by_pixel)
    motion_presence = total_scalar / (total_scalar + energy_floor)
    fit = (1.0 - residual_ratio).clamp(0, 1)
    global_confidence = (
        fit
        * support
        * scale_agreement
        * temporal_agreement
        * on_off_agreement
        * observability_global
        * motion_presence
    ).clamp(0, 1)

    observed_pseudo = observed_vector
    rotation_pseudo = expected_vector
    residual_pseudo = residual_vector
    residual_by_scale = residual.sum(dim=3)

    return RotationResidualDiagnostics(
        predicted_rotation=rotation_vector,
        predicted_rotation_flow=predicted_flow,
        on_motion_energy=on_energy if on_energy is not None else torch.zeros_like(energy),
        off_motion_energy=off_energy if off_energy is not None else torch.zeros_like(energy),
        observed_motion_energy=observed_by_scale.reshape(b, t, channels, height, width),
        explained_rotation_energy=explained.reshape(b, t, channels, height, width),
        residual_motion_energy=residual.reshape(b, t, channels, height, width),
        observed_total_energy=observed_total,
        explained_total_energy=explained_total,
        residual_total_energy=residual_total,
        residual_ratio=residual_ratio,
        observed_pseudo_flow=observed_pseudo,
        rotation_pseudo_flow=rotation_pseudo,
        residual_pseudo_flow=residual_pseudo,
        residual_energy_by_scale=residual_by_scale,
        spatial_support=support,
        scale_agreement=scale_agreement,
        temporal_agreement=temporal_agreement,
        on_off_agreement=on_off_agreement,
        observability_matrix=observability,
        observability_eigenvalues=eigenvalues,
        observability_condition=condition.unsqueeze(-1),
        axis_confidence=axis_confidence,
        global_confidence=global_confidence,
        motion_presence=motion_presence,
    )
