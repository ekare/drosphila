"""Portable normal-phone camera input and ray-space contracts.

The production contract is a single perspective RGB camera.  This module
stores calibration and timing metadata without depending on a particular
dataset or device.  Rays use the optical image convention x-right, y-down,
z-forward and are represented as [x, y, 1] for compatibility with the
project's flow geometry.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch

from .camera import CropResizeTransform
from .rotational_flow import CameraIntrinsics


@dataclass(frozen=True)
class CameraDistortion:
    """Brown-Conrady distortion coefficients in normalized camera units."""

    model: str = "none"
    k1: float = 0.0
    k2: float = 0.0
    p1: float = 0.0
    p2: float = 0.0
    k3: float = 0.0

    def __post_init__(self) -> None:
        if self.model not in {"none", "brown_conrady"}:
            raise ValueError("model must be 'none' or 'brown_conrady'")
        if self.model == "none" and any(abs(value) > 0 for value in (self.k1, self.k2, self.p1, self.p2, self.k3)):
            raise ValueError("non-zero coefficients require model='brown_conrady'")

    def as_dict(self) -> dict[str, float | str]:
        return {
            "model": self.model,
            "k1": self.k1,
            "k2": self.k2,
            "p1": self.p1,
            "p2": self.p2,
            "k3": self.k3,
        }


@dataclass(frozen=True)
class FrameTimestamp:
    timestamp_seconds: float
    frame_index: int | None = None


@dataclass(frozen=True)
class ExposureMetadata:
    exposure_seconds: float | None = None
    gain: float | None = None
    white_balance: tuple[float, float, float] | None = None


@dataclass(frozen=True)
class RollingShutterMetadata:
    line_time_seconds: float | None = None
    readout_direction: str = "top_to_bottom"

    def __post_init__(self) -> None:
        if self.readout_direction not in {"top_to_bottom", "bottom_to_top"}:
            raise ValueError("readout_direction must be top_to_bottom or bottom_to_top")


@dataclass(frozen=True)
class PhoneCameraInput:
    """Metadata accompanying one stateful phone-camera stream."""

    intrinsics: CameraIntrinsics
    timestamps: tuple[FrameTimestamp, ...]
    distortion: CameraDistortion = CameraDistortion()
    exposure: tuple[ExposureMetadata, ...] = ()
    rolling_shutter: RollingShutterMetadata | None = None
    crop_resize_history: tuple[CropResizeTransform, ...] = ()
    camera_id: str | None = None

    def validate_timestamps(self, expected_count: int | None = None) -> None:
        if expected_count is not None and len(self.timestamps) != expected_count:
            raise ValueError("timestamp count does not match frame count")
        values = [item.timestamp_seconds for item in self.timestamps]
        if any(current <= previous for previous, current in zip(values, values[1:])):
            raise ValueError("timestamps must be strictly increasing")


@dataclass(frozen=True)
class CameraRayGrid:
    """Raster-aligned camera rays and finite-validity mask."""

    rays: torch.Tensor
    valid: torch.Tensor
    intrinsics: CameraIntrinsics
    distortion: CameraDistortion

    def __post_init__(self) -> None:
        expected = (self.intrinsics.height, self.intrinsics.width, 3)
        if tuple(self.rays.shape) != expected:
            raise ValueError(f"rays must have shape H,W,3, got {tuple(self.rays.shape)}")
        if tuple(self.valid.shape) != expected[:2]:
            raise ValueError(f"valid must have shape H,W, got {tuple(self.valid.shape)}")

    @property
    def normalized_xy(self) -> torch.Tensor:
        return self.rays[..., :2]


def distort_normalized(points: torch.Tensor, distortion: CameraDistortion) -> torch.Tensor:
    """Apply Brown-Conrady distortion to [..., 2] normalized points."""

    if points.shape[-1] != 2:
        raise ValueError("points must end in two coordinates")
    if distortion.model == "none":
        return points
    x, y = points.unbind(dim=-1)
    radius2 = x.square() + y.square()
    radial = 1.0 + distortion.k1 * radius2 + distortion.k2 * radius2.square() + distortion.k3 * radius2.pow(3)
    tangential_x = 2.0 * distortion.p1 * x * y + distortion.p2 * (radius2 + 2.0 * x.square())
    tangential_y = distortion.p1 * (radius2 + 2.0 * y.square()) + 2.0 * distortion.p2 * x * y
    return torch.stack((x * radial + tangential_x, y * radial + tangential_y), dim=-1)


def undistort_normalized(points: torch.Tensor, distortion: CameraDistortion, *, iterations: int = 8) -> torch.Tensor:
    """Invert Brown-Conrady distortion with a bounded fixed-point iteration."""

    if points.shape[-1] != 2:
        raise ValueError("points must end in two coordinates")
    if iterations < 1:
        raise ValueError("iterations must be positive")
    if distortion.model == "none":
        return points
    estimate = points.clone()
    for _ in range(iterations):
        distorted = distort_normalized(estimate, distortion)
        estimate = estimate + (points - distorted)
    return estimate


def camera_ray_grid(
    intrinsics: CameraIntrinsics,
    distortion: CameraDistortion | None = None,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    undistort_iterations: int = 8,
) -> CameraRayGrid:
    """Construct an H,W,3 ray grid for arbitrary aspect ratio and principal point."""

    distortion = distortion or CameraDistortion()
    u = torch.arange(intrinsics.width, device=device, dtype=dtype)
    v = torch.arange(intrinsics.height, device=device, dtype=dtype)
    vv, uu = torch.meshgrid(v, u, indexing="ij")
    distorted = torch.stack(((uu - intrinsics.cx) / intrinsics.fx, (vv - intrinsics.cy) / intrinsics.fy), dim=-1)
    normalized = undistort_normalized(distorted, distortion, iterations=undistort_iterations)
    rays = torch.cat((normalized, torch.ones_like(normalized[..., :1])), dim=-1)
    valid = torch.isfinite(rays).all(dim=-1)
    return CameraRayGrid(rays=rays, valid=valid, intrinsics=intrinsics, distortion=distortion)


def validate_phone_frames(frames: torch.Tensor, camera: PhoneCameraInput) -> None:
    """Validate the production B,T,3,H,W frame and metadata contract."""

    if frames.ndim != 5 or frames.shape[2] != 3:
        raise ValueError("frames must have shape B,T,3,H,W")
    if tuple(frames.shape[-2:]) != (camera.intrinsics.height, camera.intrinsics.width):
        raise ValueError("frame raster does not match camera intrinsics")
    camera.validate_timestamps(frames.shape[1])
    if camera.exposure and len(camera.exposure) != frames.shape[1]:
        raise ValueError("exposure metadata count does not match frame count")


def apply_crop_resize_history(intrinsics: CameraIntrinsics, history: Sequence[CropResizeTransform]) -> CameraIntrinsics:
    """Apply a sequence of pixel-space transforms to calibration metadata."""

    result = intrinsics
    for transform in history:
        result = transform.apply(result)
    return result


def angular_to_pixel_displacement(angle_deg: float, focal_pixels: float) -> float:
    """Convert a central-ray angular displacement to a pixel displacement."""

    if focal_pixels <= 0:
        raise ValueError("focal_pixels must be positive")
    return float(focal_pixels * math.tan(math.radians(float(angle_deg))))


def pixel_to_angular_displacement(pixels: float, focal_pixels: float) -> float:
    """Convert a central-ray pixel displacement to degrees."""

    if focal_pixels <= 0:
        raise ValueError("focal_pixels must be positive")
    return float(math.degrees(math.atan(float(pixels) / focal_pixels)))


__all__ = [
    "CameraDistortion",
    "CameraRayGrid",
    "ExposureMetadata",
    "FrameTimestamp",
    "PhoneCameraInput",
    "RollingShutterMetadata",
    "apply_crop_resize_history",
    "angular_to_pixel_displacement",
    "camera_ray_grid",
    "distort_normalized",
    "pixel_to_angular_displacement",
    "undistort_normalized",
    "validate_phone_frames",
]
