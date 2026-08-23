"""Depth-conditioned rigid camera flow and scale-free translation helpers."""

from __future__ import annotations

import torch

from flyrot.diagnostics import exact_rotational_flow

from .rotational_flow import CameraIntrinsics, _hat


def rigid_camera_flow(
    rotation_vector: torch.Tensor,
    translation: torch.Tensor,
    depth: torch.Tensor,
    intrinsics: CameraIntrinsics,
) -> dict[str, torch.Tensor]:
    """Project a rigid camera transform through a depth map.

    ``translation`` is expressed in the current optical camera frame and the
    point transform is ``X_next = R @ X_current + translation``. Returned flow
    is in pixels with shape ``B,2,H,W``. The function is an oracle geometry
    primitive; RGB-only FlyRot does not infer its depth or translation inputs.
    """

    if rotation_vector.ndim != 2 or rotation_vector.shape[-1] != 3:
        raise ValueError("rotation_vector must have shape B,3")
    if translation.shape != rotation_vector.shape:
        raise ValueError("translation must have shape B,3")
    if depth.ndim != 3:
        raise ValueError("depth must have shape B,H,W")
    batch, height, width = depth.shape
    if batch != rotation_vector.shape[0] or (height, width) != (intrinsics.height, intrinsics.width):
        raise ValueError("depth and intrinsics raster/batch do not match")
    device, dtype = depth.device, depth.dtype
    u = torch.arange(width, device=device, dtype=dtype)
    v = torch.arange(height, device=device, dtype=dtype)
    vv, uu = torch.meshgrid(v, u, indexing="ij")
    x = (uu - intrinsics.cx) / intrinsics.fx
    y = (vv - intrinsics.cy) / intrinsics.fy
    rays = torch.stack((x, y, torch.ones_like(x)), dim=0)
    points = depth.unsqueeze(1) * rays.unsqueeze(0)
    rotation = torch.matrix_exp(_hat(rotation_vector))
    transformed = torch.einsum("bij,bjhw->bihw", rotation, points) + translation[:, :, None, None]
    valid = (depth > 1e-6) & (transformed[:, 2] > 1e-6) & torch.isfinite(depth)
    projected_x = transformed[:, 0] / transformed[:, 2].clamp_min(1e-6)
    projected_y = transformed[:, 1] / transformed[:, 2].clamp_min(1e-6)
    projected = torch.stack(
        (projected_x * intrinsics.fx + intrinsics.cx, projected_y * intrinsics.fy + intrinsics.cy), dim=1
    )
    original = torch.stack((uu, vv), dim=0).unsqueeze(0)
    full_flow = projected - original
    rotation_flow_normalized = exact_rotational_flow(rotation_vector, intrinsics=intrinsics)
    rotation_flow = rotation_flow_normalized * torch.tensor(
        [intrinsics.fx, intrinsics.fy], device=device, dtype=dtype
    ).view(1, 2, 1, 1)
    translation_flow = full_flow - rotation_flow
    finite = torch.isfinite(full_flow).all(dim=1)
    valid = valid & finite
    return {
        "full_flow": torch.where(valid.unsqueeze(1), full_flow, torch.zeros_like(full_flow)),
        "rotation_flow": torch.where(valid.unsqueeze(1), rotation_flow, torch.zeros_like(rotation_flow)),
        "translation_flow": torch.where(valid.unsqueeze(1), translation_flow, torch.zeros_like(translation_flow)),
        "valid": valid,
    }


def scale_free_translation_direction(translation: torch.Tensor, minimum: float = 1e-8) -> torch.Tensor:
    """Normalize translation to a direction; metric scale is intentionally discarded."""

    if translation.shape[-1] != 3:
        raise ValueError("translation must have a final dimension of 3")
    magnitude = torch.linalg.vector_norm(translation, dim=-1, keepdim=True)
    return translation / magnitude.clamp_min(minimum)


def compensate_rotational_flow(
    observed_flow: torch.Tensor,
    rotation_vector: torch.Tensor,
    intrinsics: CameraIntrinsics,
    *,
    valid: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Subtract exact rotational flow; this is an oracle/predicted-input primitive."""

    if observed_flow.ndim != 4 or observed_flow.shape[1] != 2:
        raise ValueError("observed_flow must have shape B,2,H,W")
    if observed_flow.shape[-2:] != (intrinsics.height, intrinsics.width):
        raise ValueError("observed_flow raster does not match intrinsics")
    rotation_flow = exact_rotational_flow(rotation_vector, intrinsics=intrinsics) * torch.tensor(
        [intrinsics.fx, intrinsics.fy], device=observed_flow.device, dtype=observed_flow.dtype
    ).view(1, 2, 1, 1)
    residual = observed_flow - rotation_flow
    if valid is None:
        valid = torch.isfinite(residual).all(dim=1)
    if valid.shape != residual.shape[:1] + residual.shape[-2:]:
        raise ValueError("valid must have shape B,H,W")
    residual = torch.where(valid.unsqueeze(1), residual, torch.zeros_like(residual))
    return {"rotational_flow": rotation_flow, "residual_flow": residual, "valid": valid & torch.isfinite(residual).all(dim=1)}
