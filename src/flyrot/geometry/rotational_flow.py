"""Pinhole rotational optical-flow bases in optical camera coordinates."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole intrinsics in pixel coordinates for one image raster."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.fx <= 0 or self.fy <= 0 or self.width < 1 or self.height < 1:
            raise ValueError("focal lengths and image dimensions must be positive")

    @classmethod
    def centered(cls, width: int, height: int, focal_y_over_x: float = 1.0) -> "CameraIntrinsics":
        if focal_y_over_x <= 0:
            raise ValueError("focal_y_over_x must be positive")
        return cls(
            fx=width / 2.0,
            fy=(height / 2.0) * focal_y_over_x,
            cx=(width - 1) / 2.0,
            cy=(height - 1) / 2.0,
            width=int(width),
            height=int(height),
        )

    @classmethod
    def from_mapping(cls, mapping: dict[str, float | int]) -> "CameraIntrinsics":
        return cls(
            fx=float(mapping["fx"]),
            fy=float(mapping["fy"]),
            cx=float(mapping["cx"]),
            cy=float(mapping["cy"]),
            width=int(mapping["width"]),
            height=int(mapping["height"]),
        )

    def as_dict(self) -> dict[str, float | int]:
        return {
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "width": self.width,
            "height": self.height,
        }

    def resized(self, width: int, height: int) -> "CameraIntrinsics":
        sx, sy = width / self.width, height / self.height
        return CameraIntrinsics(
            fx=self.fx * sx,
            fy=self.fy * sy,
            cx=(self.cx + 0.5) * sx - 0.5,
            cy=(self.cy + 0.5) * sy - 0.5,
            width=int(width),
            height=int(height),
        )

    def cropped(self, left: int, top: int, width: int, height: int) -> "CameraIntrinsics":
        return CameraIntrinsics(
            fx=self.fx,
            fy=self.fy,
            cx=self.cx - left,
            cy=self.cy - top,
            width=int(width),
            height=int(height),
        )


def _hat(vector: torch.Tensor) -> torch.Tensor:
    result = torch.zeros(*vector.shape[:-1], 3, 3, dtype=vector.dtype, device=vector.device)
    result[..., 0, 1] = -vector[..., 2]
    result[..., 0, 2] = vector[..., 1]
    result[..., 1, 0] = vector[..., 2]
    result[..., 1, 2] = -vector[..., 0]
    result[..., 2, 0] = -vector[..., 1]
    result[..., 2, 1] = vector[..., 0]
    return result


def normalized_camera_grid(
    intrinsics: CameraIntrinsics,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return normalized pinhole ray coordinates ``x,y`` for a raster."""

    u = torch.arange(intrinsics.width, device=device, dtype=dtype)
    v = torch.arange(intrinsics.height, device=device, dtype=dtype)
    vv, uu = torch.meshgrid(v, u, indexing="ij")
    return (uu - intrinsics.cx) / intrinsics.fx, (vv - intrinsics.cy) / intrinsics.fy


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
