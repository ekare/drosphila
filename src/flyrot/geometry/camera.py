"""Camera-raster transforms and explicit TartanAir V2 calibration metadata."""

from __future__ import annotations

from dataclasses import dataclass

from .rotational_flow import CameraIntrinsics


@dataclass(frozen=True)
class CropResizeTransform:
    """A pixel-space crop followed by an anisotropic raster resize."""

    left: int
    top: int
    crop_width: int
    crop_height: int
    output_width: int
    output_height: int

    def __post_init__(self) -> None:
        if min(self.left, self.top) < 0:
            raise ValueError("crop offsets must be non-negative")
        if min(self.crop_width, self.crop_height, self.output_width, self.output_height) < 1:
            raise ValueError("crop and output dimensions must be positive")

    def apply(self, intrinsics: CameraIntrinsics) -> CameraIntrinsics:
        """Transform source intrinsics into the output raster."""

        cropped = intrinsics.cropped(self.left, self.top, self.crop_width, self.crop_height)
        return cropped.resized(self.output_width, self.output_height)


def tartanair_v2_lcam_front_intrinsics(width: int = 640, height: int = 640) -> CameraIntrinsics:
    """Return the documented TartanAir V2 lcam-front calibration assumption.

    The public V2 convention is commonly written as ``f=320`` and principal
    point ``(320,320)`` for a 640x640 raster. This helper makes that assumption
    explicit; it is not a claim that every downloaded local camera manifest has
    been independently verified.
    """

    scale_x, scale_y = width / 640.0, height / 640.0
    return CameraIntrinsics(
        fx=320.0 * scale_x,
        fy=320.0 * scale_y,
        cx=320.0 * scale_x,
        cy=320.0 * scale_y,
        width=int(width),
        height=int(height),
    )
