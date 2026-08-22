"""Validity-aware directional residual and rotation-evidence diagnostics.

The direction-cell output is native non-negative direction evidence, not
optical flow. This module compares its direction with the exact perspective
direction induced by a camera rotation, while keeping every residual in
evidence space. Pseudo-flow fields are visualization-only.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from flyrot.geometry.rotational_flow import (
    CameraIntrinsics,
    _hat,
    normalized_camera_grid,
    rotational_flow_basis,
)


def _direction_vectors(directions: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    angles = torch.arange(directions, device=device, dtype=dtype) * (2.0 * torch.pi / directions)
    return torch.stack((angles.cos(), angles.sin()), dim=1)


def _safe_depth(depth: torch.Tensor, minimum: float = 1e-6) -> torch.Tensor:
    sign = torch.where(depth < 0, -torch.ones_like(depth), torch.ones_like(depth))
    return torch.where(depth.abs() < minimum, sign * minimum, depth)


def exact_rotational_flow(
    rotation_vector: torch.Tensor,
    height: int | None = None,
    width: int | None = None,
    focal_y_over_x: float = 1.0,
    *,
    intrinsics: CameraIntrinsics | None = None,
) -> torch.Tensor:
    """Project normalized camera rays after an active SO(3) rotation.

    The result has shape ``(..., 2, H, W)`` and is a displacement in
    normalized camera-ray coordinates. Passing ``intrinsics`` enables a
    shifted principal point and non-square focal lengths. The legacy centered
    ``height, width, focal_y_over_x`` path remains equivalent to a centered
    :class:`CameraIntrinsics` instance.
    """

    if rotation_vector.ndim < 1 or rotation_vector.shape[-1] != 3:
        raise ValueError(f"expected (..., 3) rotation vectors, got {tuple(rotation_vector.shape)}")
    if intrinsics is None:
        if height is None or width is None:
            raise ValueError("height and width are required without intrinsics")
        intrinsics = CameraIntrinsics.centered(width, height, focal_y_over_x)
    elif (height is not None and height != intrinsics.height) or (width is not None and width != intrinsics.width):
        raise ValueError("height/width must match the supplied intrinsics")
    device, dtype = rotation_vector.device, rotation_vector.dtype
    x, y = normalized_camera_grid(intrinsics, device=device, dtype=dtype)
    rays = torch.stack((x, y, torch.ones_like(x)), dim=-1)
    rotation = torch.matrix_exp(_hat(rotation_vector))
    rotated = torch.einsum("...ij,hwj->...hwi", rotation, rays)
    projected = rotated[..., :2] / _safe_depth(rotated[..., 2:3])
    flow = projected - rays[..., :2]
    return flow.movedim(-1, -3)


def _basis_for_intrinsics(
    intrinsics: CameraIntrinsics,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    x, y = normalized_camera_grid(intrinsics, device=device, dtype=dtype)
    return rotational_flow_basis(x, y)


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
    normalizer = torch.log(torch.tensor(float(h * w), device=flat.device, dtype=flat.dtype)).clamp_min(1.0)
    support = entropy / normalizer
    return torch.where(total.squeeze(-1) > 1e-8, support, torch.zeros_like(support)).unsqueeze(-1)


def _weighted_scale_mean(
    evidence: torch.Tensor,
    strength: torch.Tensor,
    valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Aggregate ``B,T,S,3`` evidence and return its validity."""

    weights = strength * valid.to(strength.dtype)
    total = weights.sum(dim=2, keepdim=True)
    mean = (evidence * weights.unsqueeze(-1)).sum(dim=2) / total.squeeze(2).unsqueeze(-1).clamp_min(1e-8)
    return mean, total.squeeze(2) > 1e-8


def estimate_rotation_evidence(
    energy: torch.Tensor,
    valid: torch.Tensor,
    intrinsics: CameraIntrinsics | None = None,
    *,
    focal_y_over_x: float = 1.0,
    directions: int = 8,
    energy_floor: float = 1e-3,
    condition_limit: float = 1e8,
) -> dict[str, torch.Tensor]:
    """Estimate geometry-level rotation evidence for every scale.

    A direction-energy vector is projected through the same rotational-flow
    Jacobian used by the model's rotation evidence. Weighted least squares
    returns one 3-vector and information matrix per ``B,T,S`` state. The
    returned validity masks reject low-energy or numerically bad states.
    """

    if energy.ndim != 5 or valid.shape != energy.shape:
        raise ValueError("energy and valid must both have shape B,T,C,H,W")
    if directions < 2 or energy.shape[2] % directions:
        raise ValueError("energy channel count must be divisible by directions")
    if energy_floor <= 0:
        raise ValueError("energy_floor must be positive")
    b, t, channels, height, width = energy.shape
    scales = channels // directions
    if intrinsics is None:
        intrinsics = CameraIntrinsics.centered(width, height, focal_y_over_x)
    if (intrinsics.height, intrinsics.width) != (height, width):
        raise ValueError("intrinsics raster does not match energy raster")
    device, dtype = energy.device, energy.dtype
    basis = _basis_for_intrinsics(intrinsics, device=device, dtype=dtype)
    basis_hw = basis.permute(2, 3, 1, 0).contiguous()  # H,W,2,3
    direction_vectors = _direction_vectors(directions, device=device, dtype=dtype)
    weighted = energy.clamp_min(0) * valid.to(dtype=dtype).clamp_min(0)
    by_scale = weighted.reshape(b, t, scales, directions, height, width)
    motion = torch.einsum("btsdhw,dp->btsphw", by_scale, direction_vectors)
    pixel_weight = by_scale.sum(dim=3)
    total = pixel_weight.sum(dim=(3, 4))
    normal = torch.einsum("hwij,btshw,hwik->btsjk", basis_hw, pixel_weight, basis_hw)
    motion_hw = motion.permute(0, 1, 2, 4, 5, 3).contiguous()
    rhs = torch.einsum("hwij,btshw,btshwi->btsj", basis_hw, pixel_weight, motion_hw)
    normalized_info = normal / total.unsqueeze(-1).unsqueeze(-1).clamp_min(1e-8)
    normalized_rhs = rhs / total.unsqueeze(-1).clamp_min(1e-8)
    regularizer = torch.eye(3, device=device, dtype=dtype).view(1, 1, 1, 3, 3) * 1e-5
    information = normalized_info
    evidence = torch.linalg.solve(information + regularizer, normalized_rhs.unsqueeze(-1)).squeeze(-1)
    eigenvalues = torch.linalg.eigvalsh(information + regularizer).clamp_min(0)
    condition = eigenvalues[..., -1] / eigenvalues[..., 0].clamp_min(1e-8)
    evidence_strength = total / (total + energy_floor)
    evidence_valid = (total > energy_floor) & torch.isfinite(evidence).all(dim=-1) & (condition < condition_limit)
    aggregate, aggregate_valid = _weighted_scale_mean(evidence, evidence_strength, evidence_valid)
    return {
        "rotation_evidence": evidence,
        "information_matrix": information,
        "evidence_strength": evidence_strength,
        "evidence_valid": evidence_valid,
        "evidence_eigenvalues": eigenvalues,
        "evidence_condition": condition,
        "aggregate_rotation_evidence": aggregate,
        "aggregate_valid": aggregate_valid,
        "total_energy": total,
    }


def _agreement_to_reference(
    evidence: torch.Tensor,
    strength: torch.Tensor,
    valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compare valid scale evidence in geometry space, not pixel pseudo-flow."""

    weights = strength * valid.to(strength.dtype)
    count = valid.sum(dim=2)
    reference = (evidence * weights.unsqueeze(-1)).sum(dim=2) / weights.sum(dim=2).unsqueeze(-1).clamp_min(1e-8)
    reference_norm = torch.linalg.vector_norm(reference, dim=-1)
    unit = evidence / torch.linalg.vector_norm(evidence, dim=-1, keepdim=True).clamp_min(1e-8)
    ref_unit = reference / reference_norm.unsqueeze(-1).clamp_min(1e-8)
    cosine = (unit * ref_unit.unsqueeze(2)).sum(dim=-1).clamp(0, 1)
    score = (cosine * weights).sum(dim=2) / weights.sum(dim=2).clamp_min(1e-8)
    agreement_valid = (count >= 2) & (reference_norm > 1e-8)
    return torch.where(agreement_valid, score, torch.full_like(score, torch.nan)), agreement_valid, count


@dataclass
class RotationResidualDiagnostics:
    """Structured outputs for one batch of causal model states."""

    predicted_rotation: torch.Tensor
    predicted_rotation_flow: torch.Tensor
    on_motion_energy: torch.Tensor
    off_motion_energy: torch.Tensor
    observed_motion_energy: torch.Tensor
    rotation_direction_compatible_energy: torch.Tensor
    directional_residual_motion_energy: torch.Tensor
    observed_total_energy: torch.Tensor
    rotation_direction_compatible_total_energy: torch.Tensor
    directional_residual_total_energy: torch.Tensor
    directional_residual_ratio: torch.Tensor
    directional_fit_score: torch.Tensor
    observed_pseudo_flow: torch.Tensor
    rotation_pseudo_flow: torch.Tensor
    residual_pseudo_flow: torch.Tensor
    residual_energy_by_scale: torch.Tensor
    spatial_support_score: torch.Tensor
    spatial_support_valid: torch.Tensor
    scale_rotation_evidence: torch.Tensor
    scale_evidence_strength: torch.Tensor
    scale_evidence_valid: torch.Tensor
    scale_agreement_score: torch.Tensor
    scale_agreement_valid: torch.Tensor
    active_scale_count: torch.Tensor
    temporal_rotation_evidence: torch.Tensor
    temporal_agreement_score: torch.Tensor
    temporal_agreement_valid: torch.Tensor
    valid_motion_state_count: torch.Tensor
    valid_temporal_pair_count: torch.Tensor
    on_rotation_evidence: torch.Tensor
    off_rotation_evidence: torch.Tensor
    on_off_agreement_score: torch.Tensor
    on_off_agreement_valid: torch.Tensor
    on_energy_total: torch.Tensor
    off_energy_total: torch.Tensor
    on_off_balance_score: torch.Tensor
    observability_matrix: torch.Tensor
    observability_eigenvalues: torch.Tensor
    observability_condition: torch.Tensor
    observability_isotropy: torch.Tensor
    observability_covariance: torch.Tensor
    observability_valid: torch.Tensor
    observability_score: torch.Tensor
    axis_variance: torch.Tensor
    axis_information: torch.Tensor
    axis_confidence: torch.Tensor
    motion_presence_score: torch.Tensor
    motion_presence_valid: torch.Tensor
    global_reliability: torch.Tensor
    global_reliability_valid: torch.Tensor

    @property
    def residual_ratio(self) -> torch.Tensor:
        return self.directional_residual_ratio

    @property
    def scale_agreement(self) -> torch.Tensor:
        return self.scale_agreement_score

    @property
    def temporal_agreement(self) -> torch.Tensor:
        return self.temporal_agreement_score

    @property
    def on_off_agreement(self) -> torch.Tensor:
        return self.on_off_agreement_score

    @property
    def global_confidence(self) -> torch.Tensor:
        return self.global_reliability

    @property
    def explained_total_energy(self) -> torch.Tensor:
        return self.rotation_direction_compatible_total_energy

    @property
    def residual_total_energy(self) -> torch.Tensor:
        return self.directional_residual_total_energy

    def as_dict(self) -> dict[str, torch.Tensor]:
        """Return v0.3 names plus one-release backward-compatible aliases."""

        result = {key: value for key, value in self.__dict__.items() if isinstance(value, torch.Tensor)}
        result.update(
            {
                # Deprecated v0.2 aliases. They are aliases, not old semantics.
                "residual_ratio": self.directional_residual_ratio,
                "explained_rotation_energy": self.rotation_direction_compatible_energy,
                "residual_motion_energy": self.directional_residual_motion_energy,
                "explained_total_energy": self.rotation_direction_compatible_total_energy,
                "residual_total_energy": self.directional_residual_total_energy,
                "spatial_support": self.spatial_support_score,
                "scale_agreement": self.scale_agreement_score,
                "temporal_agreement": self.temporal_agreement_score,
                "on_off_agreement": self.on_off_agreement_score,
                "motion_presence": self.motion_presence_score,
                "global_confidence": self.global_reliability,
            }
        )
        return result


def _directional_decomposition(
    rotation_vector: torch.Tensor,
    energy: torch.Tensor,
    valid: torch.Tensor,
    *,
    intrinsics: CameraIntrinsics,
    directions: int,
) -> dict[str, torch.Tensor]:
    b, t, channels, height, width = energy.shape
    scales = channels // directions
    observed = energy.clamp_min(0) * valid.to(dtype=energy.dtype).clamp_min(0)
    observed_by_scale = observed.reshape(b, t, scales, directions, height, width)
    predicted_flow = exact_rotational_flow(rotation_vector, intrinsics=intrinsics)
    predicted_magnitude = torch.linalg.vector_norm(predicted_flow, dim=2, keepdim=True)
    flow_unit = predicted_flow / predicted_magnitude.clamp_min(1e-8)
    vectors = _direction_vectors(directions, device=energy.device, dtype=energy.dtype)
    compatibility = torch.einsum("dp,btphw->btdhw", vectors, flow_unit).clamp_min(0)
    compatibility = compatibility * (predicted_magnitude > 1e-8)
    compatibility = compatibility.unsqueeze(2).expand(b, t, scales, directions, height, width)
    explained = observed_by_scale * compatibility
    residual = (observed_by_scale - explained).clamp_min(0)
    total_by_pixel = observed_by_scale.sum(dim=(2, 3))
    observed_total = total_by_pixel.unsqueeze(2)
    explained_total = explained.sum(dim=(2, 3)).unsqueeze(2)
    residual_total = residual.sum(dim=(2, 3)).unsqueeze(2)
    observed_scalar = total_by_pixel.sum(dim=(2, 3), keepdim=False).unsqueeze(-1)
    residual_scalar = residual_total.sum(dim=(3, 4))
    directional_ratio = residual_scalar / observed_scalar.clamp_min(1e-8)
    directional_ratio = torch.where(observed_scalar > 1e-8, directional_ratio, torch.ones_like(directional_ratio)).clamp(0, 1)
    observed_vector = torch.einsum("btsdhw,dp->btphw", observed_by_scale, vectors)
    explained_vector = torch.einsum("btsdhw,dp->btphw", explained, vectors)
    residual_vector = torch.einsum("btsdhw,dp->btphw", residual, vectors)
    return {
        "predicted_flow": predicted_flow,
        "observed_by_scale": observed_by_scale,
        "observed_total": observed_total,
        "explained": explained,
        "explained_total": explained_total,
        "residual": residual,
        "residual_total": residual_total,
        "directional_ratio": directional_ratio,
        "directional_fit": (1.0 - directional_ratio).clamp(0, 1),
        "observed_vector": observed_vector,
        "explained_vector": explained_vector,
        "residual_vector": residual_vector,
        "total_by_pixel": total_by_pixel,
        "observed_scalar": observed_scalar,
    }


def _channel_evidence(
    channel_energy: torch.Tensor,
    valid: torch.Tensor,
    *,
    intrinsics: CameraIntrinsics,
    energy_floor: float,
) -> dict[str, torch.Tensor]:
    return estimate_rotation_evidence(channel_energy, valid, intrinsics, directions=8, energy_floor=energy_floor)


def _temporal_scores(
    evidence: torch.Tensor,
    valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    b, t, _ = evidence.shape
    score = torch.full((b, t, 1), torch.nan, device=evidence.device, dtype=evidence.dtype)
    pair_valid = torch.zeros((b, t, 1), device=evidence.device, dtype=torch.bool)
    if t > 1:
        cosine = _cosine(evidence[:, 1:], evidence[:, :-1], dim=-1).clamp(0, 1)
        current_valid = valid[:, 1:] & valid[:, :-1]
        current_score = torch.where(
            current_valid.unsqueeze(-1),
            cosine.unsqueeze(-1),
            torch.full_like(cosine.unsqueeze(-1), torch.nan),
        )
        score[:, 1:] = current_score
        pair_valid[:, 1:] = current_valid.unsqueeze(-1)
    motion_count = valid.sum(dim=1, keepdim=True)
    pair_count = pair_valid.squeeze(-1).sum(dim=1, keepdim=True)
    return score, pair_valid, motion_count, pair_count


def _validity_aware_reliability(
    directional_fit: torch.Tensor,
    directional_valid: torch.Tensor,
    spatial_support: torch.Tensor,
    spatial_valid: torch.Tensor,
    scale_score: torch.Tensor,
    scale_valid: torch.Tensor,
    temporal_score: torch.Tensor,
    temporal_valid: torch.Tensor,
    on_off_score: torch.Tensor,
    on_off_valid: torch.Tensor,
    observability_score: torch.Tensor,
    observability_valid: torch.Tensor,
    motion_presence: torch.Tensor,
    balance: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    components = (directional_fit, spatial_support, scale_score, temporal_score, on_off_score, observability_score)
    masks = (directional_valid, spatial_valid, scale_valid, temporal_valid, on_off_valid, observability_valid)
    weights = (2.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    log_sum = torch.zeros_like(directional_fit)
    weight_sum = torch.zeros_like(directional_fit)
    for component, mask, weight in zip(components, masks, weights):
        safe = torch.where(mask, component.clamp_min(1e-6), torch.ones_like(component))
        log_sum = log_sum + mask.to(component.dtype) * weight * safe.clamp_min(1e-6).log()
        weight_sum = weight_sum + mask.to(component.dtype) * weight
    valid = weight_sum > 0
    core = torch.exp(log_sum / weight_sum.clamp_min(1e-8))
    reliability = (core * motion_presence * balance).clamp(0, 1)
    reliability = torch.where(valid, reliability, torch.zeros_like(reliability))
    return reliability, valid


def build_rotation_residual_diagnostics(
    rotation_vector: torch.Tensor,
    energy: torch.Tensor,
    valid: torch.Tensor,
    *,
    on_energy: torch.Tensor | None = None,
    off_energy: torch.Tensor | None = None,
    focal_y_over_x: float = 1.0,
    intrinsics: CameraIntrinsics | None = None,
    energy_floor: float = 1e-3,
) -> RotationResidualDiagnostics:
    """Build directional residuals and validity-aware reliability fields.

    ``directional_residual_ratio`` measures the share of observed local
    direction evidence that is not directionally compatible with the exact
    predicted rotation field. It is not physical motion energy or optical
    flow. ``residual_ratio`` remains only as a deprecated alias.
    """

    if energy.ndim != 5 or valid.shape != energy.shape:
        raise ValueError("energy and valid must both have shape B,T,C,H,W")
    if rotation_vector.ndim != 3 or rotation_vector.shape[:2] != energy.shape[:2]:
        raise ValueError("rotation_vector must have shape B,T,3 matching energy")
    b, t, channels, height, width = energy.shape
    if channels % 8 != 0:
        raise ValueError("direction energy channel count must be a multiple of 8")
    if on_energy is not None and on_energy.shape != energy.shape:
        raise ValueError("on_energy must match energy shape")
    if off_energy is not None and off_energy.shape != energy.shape:
        raise ValueError("off_energy must match energy shape")
    if intrinsics is None:
        intrinsics = CameraIntrinsics.centered(width, height, focal_y_over_x)
    if (intrinsics.height, intrinsics.width) != (height, width):
        raise ValueError("intrinsics raster does not match energy raster")

    decomposition = _directional_decomposition(rotation_vector, energy, valid, intrinsics=intrinsics, directions=8)
    evidence = estimate_rotation_evidence(energy, valid, intrinsics, directions=8, energy_floor=energy_floor)
    scale_score, scale_valid, active_scales = _agreement_to_reference(
        evidence["rotation_evidence"], evidence["evidence_strength"], evidence["evidence_valid"]
    )
    temporal_evidence = evidence["aggregate_rotation_evidence"]
    temporal_valid_state = evidence["aggregate_valid"]
    temporal_score, temporal_valid, valid_motion_count, valid_pair_count = _temporal_scores(
        temporal_evidence, temporal_valid_state
    )

    if on_energy is None or off_energy is None:
        on_energy_value = torch.zeros_like(energy) if on_energy is None else on_energy
        off_energy_value = torch.zeros_like(energy) if off_energy is None else off_energy
        on_evidence = _channel_evidence(on_energy_value, valid, intrinsics=intrinsics, energy_floor=energy_floor)
        off_evidence = _channel_evidence(off_energy_value, valid, intrinsics=intrinsics, energy_floor=energy_floor)
        on_off_score = torch.full((b, t, 1), torch.nan, device=energy.device, dtype=energy.dtype)
        on_off_valid = torch.zeros((b, t, 1), device=energy.device, dtype=torch.bool)
    else:
        on_energy_value, off_energy_value = on_energy, off_energy
        on_evidence = _channel_evidence(on_energy, valid, intrinsics=intrinsics, energy_floor=energy_floor)
        off_evidence = _channel_evidence(off_energy, valid, intrinsics=intrinsics, energy_floor=energy_floor)
        on_mean, on_valid = _weighted_scale_mean(
            on_evidence["rotation_evidence"], on_evidence["evidence_strength"], on_evidence["evidence_valid"]
        )
        off_mean, off_valid = _weighted_scale_mean(
            off_evidence["rotation_evidence"], off_evidence["evidence_strength"], off_evidence["evidence_valid"]
        )
        on_total = on_evidence["total_energy"].sum(dim=2)
        off_total = off_evidence["total_energy"].sum(dim=2)
        on_off_valid = (on_valid & off_valid & (on_total > energy_floor) & (off_total > energy_floor)).unsqueeze(-1)
        on_off_score = _cosine(on_mean, off_mean, dim=-1).clamp(0, 1).unsqueeze(-1)
        on_off_score = torch.where(on_off_valid, on_off_score, torch.full_like(on_off_score, torch.nan))
    on_total = on_evidence["total_energy"].sum(dim=2).unsqueeze(-1)
    off_total = off_evidence["total_energy"].sum(dim=2).unsqueeze(-1)
    balance = 2.0 * torch.minimum(on_total, off_total) / (on_total + off_total + 1e-8)
    balance = torch.where(on_total + off_total > energy_floor, balance, torch.zeros_like(balance)).clamp(0, 1)

    total_scalar = decomposition["observed_scalar"]
    info_weights = evidence["evidence_strength"] * evidence["evidence_valid"].to(energy.dtype)
    combined_info = (evidence["information_matrix"] * info_weights.unsqueeze(-1).unsqueeze(-1)).sum(dim=2)
    combined_info = combined_info / info_weights.sum(dim=2).unsqueeze(-1).unsqueeze(-1).clamp_min(1e-8)
    reg = torch.eye(3, device=energy.device, dtype=energy.dtype).view(1, 1, 3, 3) * 1e-4
    covariance = torch.linalg.inv(combined_info + reg)
    observability_eigenvalues = torch.linalg.eigvalsh(combined_info + reg).clamp_min(0)
    observability_condition = observability_eigenvalues[..., -1] / observability_eigenvalues[..., 0].clamp_min(1e-8)
    observability_isotropy = 3.0 * observability_eigenvalues[..., 0] / observability_eigenvalues.sum(dim=-1).clamp_min(1e-8)
    observability_valid = (total_scalar.squeeze(-1) > energy_floor) & (observability_condition < 1e8)
    observability_score = torch.where(
        observability_valid,
        observability_isotropy,
        torch.full_like(observability_isotropy, torch.nan),
    ).unsqueeze(-1)
    axis_variance = covariance.diagonal(dim1=-2, dim2=-1)
    axis_information = combined_info.diagonal(dim1=-2, dim2=-1).clamp_min(0)
    axis_confidence = axis_information / axis_information.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    axis_confidence = torch.where(observability_valid.unsqueeze(-1), axis_confidence, torch.zeros_like(axis_confidence))

    spatial_support = _spatial_entropy_support(decomposition["total_by_pixel"])
    spatial_valid = total_scalar > energy_floor
    motion_presence = total_scalar / (total_scalar + energy_floor)
    motion_valid = total_scalar > energy_floor
    reliability, reliability_valid = _validity_aware_reliability(
        decomposition["directional_fit"],
        motion_valid,
        spatial_support,
        spatial_valid,
        scale_score.unsqueeze(-1),
        scale_valid.unsqueeze(-1),
        temporal_score,
        temporal_valid,
        on_off_score,
        on_off_valid,
        observability_score,
        observability_valid.unsqueeze(-1),
        motion_presence,
        balance,
    )
    return RotationResidualDiagnostics(
        predicted_rotation=rotation_vector,
        predicted_rotation_flow=decomposition["predicted_flow"],
        on_motion_energy=on_energy_value,
        off_motion_energy=off_energy_value,
        observed_motion_energy=decomposition["observed_by_scale"].reshape(b, t, channels, height, width),
        rotation_direction_compatible_energy=decomposition["explained"].reshape(b, t, channels, height, width),
        directional_residual_motion_energy=decomposition["residual"].reshape(b, t, channels, height, width),
        observed_total_energy=decomposition["observed_total"],
        rotation_direction_compatible_total_energy=decomposition["explained_total"],
        directional_residual_total_energy=decomposition["residual_total"],
        directional_residual_ratio=decomposition["directional_ratio"],
        directional_fit_score=decomposition["directional_fit"],
        observed_pseudo_flow=decomposition["observed_vector"],
        rotation_pseudo_flow=decomposition["explained_vector"],
        residual_pseudo_flow=decomposition["residual_vector"],
        residual_energy_by_scale=decomposition["residual"].sum(dim=3),
        spatial_support_score=spatial_support,
        spatial_support_valid=spatial_valid,
        scale_rotation_evidence=evidence["rotation_evidence"],
        scale_evidence_strength=evidence["evidence_strength"],
        scale_evidence_valid=evidence["evidence_valid"],
        scale_agreement_score=scale_score.unsqueeze(-1),
        scale_agreement_valid=scale_valid.unsqueeze(-1),
        active_scale_count=active_scales.unsqueeze(-1),
        temporal_rotation_evidence=temporal_evidence,
        temporal_agreement_score=temporal_score,
        temporal_agreement_valid=temporal_valid,
        valid_motion_state_count=valid_motion_count,
        valid_temporal_pair_count=valid_pair_count,
        on_rotation_evidence=on_evidence["aggregate_rotation_evidence"],
        off_rotation_evidence=off_evidence["aggregate_rotation_evidence"],
        on_off_agreement_score=on_off_score,
        on_off_agreement_valid=on_off_valid,
        on_energy_total=on_total,
        off_energy_total=off_total,
        on_off_balance_score=balance,
        observability_matrix=combined_info,
        observability_eigenvalues=observability_eigenvalues,
        observability_condition=observability_condition.unsqueeze(-1),
        observability_isotropy=observability_isotropy.unsqueeze(-1),
        observability_covariance=covariance,
        observability_valid=observability_valid.unsqueeze(-1),
        observability_score=observability_score,
        axis_variance=axis_variance,
        axis_information=axis_information,
        axis_confidence=axis_confidence,
        motion_presence_score=motion_presence,
        motion_presence_valid=motion_valid,
        global_reliability=reliability,
        global_reliability_valid=reliability_valid,
    )
