"""Deterministic T4/T5-inspired cell stimuli and engineering metrics.

The implementation intentionally makes no biological-equivalence claim.  It
only tests whether the existing ON/OFF delayed-correlation bank has the
direction, polarity, speed, and border behavior required by the project
contract.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch.nn import functional as F

from flyrot.models.direction_cells import DIRECTION_NAMES, DirectionCellBank
from flyrot.models.photoreceptor import Photoreceptor


STIMULUS_CLASSES = (
    "stationary_uniform",
    "brightness_increase",
    "brightness_decrease",
    "flicker",
    "moving_bright_edge",
    "moving_dark_edge",
    "moving_bright_bar",
    "moving_dark_bar",
    "moving_sinusoidal_grating",
    "contrast_reversed_edge",
    "moving_patch",
    "aperture_limited_edge",
)
ANGLES_DEG = tuple(range(0, 360, 45))
DISPLACEMENTS = (0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 16.0)
CONTRASTS = {"low": 0.25, "medium": 0.55, "high": 0.9}


@dataclass(frozen=True)
class StimulusSpec:
    kind: str
    angle_deg: float = 0.0
    displacement: float = 4.0
    contrast_name: str = "medium"
    height: int = 48
    width: int = 64
    steps: int = 5
    gamma: float = 1.0
    brightness_offset: float = 0.0
    noise_std: float = 0.0
    blur_sigma: float = 0.0
    seed: int = 0

    @property
    def direction_index(self) -> int:
        return int(round(self.angle_deg / 45.0)) % 8


def _blur(frame: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return frame
    radius = max(1, int(math.ceil(3.0 * sigma)))
    coordinates = torch.arange(-radius, radius + 1, dtype=frame.dtype, device=frame.device)
    kernel = torch.exp(-0.5 * (coordinates / sigma).square())
    kernel = kernel / kernel.sum()
    frame = F.conv2d(frame[None, None], kernel.view(1, 1, 1, -1), padding=(0, radius))
    frame = F.conv2d(frame, kernel.view(1, 1, -1, 1), padding=(radius, 0))
    return frame[0, 0]


def _render_base(spec: StimulusSpec, time_index: int) -> torch.Tensor:
    if spec.kind not in STIMULUS_CLASSES:
        raise ValueError(f"unknown stimulus class: {spec.kind}")
    y = torch.arange(spec.height, dtype=torch.float32)
    x = torch.arange(spec.width, dtype=torch.float32)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    cx = (spec.width - 1) / 2.0
    cy = (spec.height - 1) / 2.0
    radians = math.radians(spec.angle_deg)
    direction = torch.tensor([math.cos(radians), math.sin(radians)])
    normal = torch.tensor([-direction[1], direction[0]])
    position = torch.tensor([cx, cy]) + direction * (time_index * spec.displacement)
    signed_normal = (xx - position[0]) * normal[0] + (yy - position[1]) * normal[1]
    signed_motion = (xx - position[0]) * direction[0] + (yy - position[1]) * direction[1]
    contrast = CONTRASTS[spec.contrast_name]
    background = torch.full_like(xx, 0.1)
    foreground = background + 0.85 * contrast
    bright = spec.kind in {"moving_bright_edge", "moving_bright_bar", "moving_patch", "aperture_limited_edge"}
    dark = spec.kind in {"moving_dark_edge", "moving_dark_bar"}
    if spec.kind == "stationary_uniform":
        frame = torch.full_like(xx, 0.5)
    elif spec.kind == "brightness_increase":
        frame = torch.full_like(xx, 0.25 + (0.45 if time_index >= 2 else 0.0))
    elif spec.kind == "brightness_decrease":
        frame = torch.full_like(xx, 0.75 - (0.45 if time_index >= 2 else 0.0))
    elif spec.kind == "flicker":
        frame = torch.full_like(xx, 0.25 if time_index % 2 == 0 else 0.75)
    elif spec.kind in {"moving_bright_edge", "moving_dark_edge", "contrast_reversed_edge", "aperture_limited_edge"}:
        if spec.kind == "contrast_reversed_edge":
            bright = time_index % 2 == 0
            dark = not bright
        # A translating edge is oriented perpendicular to its motion; its
        # boundary therefore uses the coordinate along the motion vector.
        edge = signed_motion >= 0
        frame = torch.where(edge, foreground, background)
        if dark:
            frame = torch.where(edge, torch.full_like(xx, 0.9 - 0.8 * contrast), torch.full_like(xx, 0.9))
        if spec.kind == "aperture_limited_edge":
            radius = min(spec.height, spec.width) * 0.38
            aperture = (xx - cx).square() + (yy - cy).square() <= radius * radius
            frame = torch.where(aperture, frame, torch.full_like(xx, 0.5))
    elif spec.kind in {"moving_bright_bar", "moving_dark_bar"}:
        bar = signed_motion.abs() <= max(2.0, min(spec.height, spec.width) * 0.12)
        if bright:
            frame = torch.where(bar, foreground, background)
        else:
            frame = torch.where(bar, torch.full_like(xx, 0.9 - 0.8 * contrast), torch.full_like(xx, 0.9))
    elif spec.kind == "moving_sinusoidal_grating":
        phase = 2.0 * math.pi * (signed_motion / max(8.0, min(spec.height, spec.width) * 0.22))
        frame = 0.5 + 0.42 * contrast * torch.cos(phase)
    elif spec.kind == "moving_patch":
        radius = min(spec.height, spec.width) * 0.16
        patch = signed_normal.square() + signed_motion.square() <= radius * radius
        frame = torch.where(patch, foreground, background)
    else:
        raise AssertionError(spec.kind)
    if spec.gamma != 1.0:
        frame = frame.clamp(0.0, 1.0).pow(1.0 / spec.gamma)
    frame = frame + spec.brightness_offset
    if spec.noise_std > 0:
        generator = torch.Generator().manual_seed(spec.seed + time_index * 7919)
        frame = frame + torch.randn(frame.shape, generator=generator) * spec.noise_std
    return _blur(frame.clamp(0.0, 1.0), spec.blur_sigma).clamp(0.0, 1.0)


def make_stimulus(spec: StimulusSpec) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``T,1,H,W`` frames and a central valid mask.

    The mask excludes pixels whose translated stimulus or direction-cell
    neighborhood can touch a raster border.  Border-crossing variants are
    still generated, but their border region is measured separately.
    """

    frames = torch.stack([_render_base(spec, index) for index in range(spec.steps)])[:, None]
    margin = int(math.ceil(abs(spec.displacement) * (spec.steps - 1))) + 18
    valid = torch.zeros((1, spec.height, spec.width), dtype=torch.float32)
    if 2 * margin < min(spec.height, spec.width):
        valid[:, margin : spec.height - margin, margin : spec.width - margin] = 1.0
    return frames, valid


def _mean_response(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weighted = values * mask
    return weighted.sum(dim=(-1, -2)) / mask.sum(dim=(-1, -2)).clamp_min(1.0)


def evaluate_bank(
    frames: torch.Tensor,
    valid_spatial: torch.Tensor,
    *,
    scales: tuple[int, ...],
    device: torch.device | str | None = None,
) -> dict[str, torch.Tensor | float | int | list[float]]:
    """Evaluate one sequence, preserving ON and OFF responses separately."""

    if frames.ndim != 4:
        raise ValueError("frames must have shape T,1,H,W")
    target_device = torch.device(device) if device is not None else frames.device
    photoreceptor = Photoreceptor().to(target_device)
    bank = DirectionCellBank(scales=scales).to(target_device)
    frames = frames.to(target_device)
    valid_spatial = valid_spatial.to(target_device)
    on, off = photoreceptor(frames[None])
    energy, valid, on_energy, off_energy = bank(on, off, return_components=True)
    # Bank output is B,steps,scale*direction,H,W.  Aggregate scales only after
    # masking, so each direction keeps its own valid-pixel accounting.
    b, steps, channels, height, width = energy.shape
    on_by_direction = on_energy.reshape(b, steps, len(scales), 8, height, width)
    off_by_direction = off_energy.reshape(b, steps, len(scales), 8, height, width)
    valid_by_direction = valid.reshape(b, steps, len(scales), 8, height, width)
    spatial = valid_spatial[None, None, None]
    on_response = _mean_response(on_by_direction, valid_by_direction * spatial).mean(dim=(1, 2))[0]
    off_response = _mean_response(off_by_direction, valid_by_direction * spatial).mean(dim=(1, 2))[0]
    response = on_response + off_response
    preferred = int(response.argmax().item())
    null = (preferred + 4) % 8
    pref = float(response[preferred])
    null_value = float(response[null])
    dsi = (pref - null_value) / (pref + null_value + 1e-8)
    return {
        "on_response": on_response,
        "off_response": off_response,
        "response": response,
        "preferred_index": preferred,
        "null_index": null,
        "preferred_response": pref,
        "null_response": null_value,
        "preferred_null_ratio": pref / max(null_value, 1e-8),
        "dsi": dsi,
        "valid_fraction": float((valid_by_direction * spatial).mean()),
    }


def direction_accuracy(spec: StimulusSpec, result: dict[str, object]) -> float:
    if spec.kind not in {"moving_bright_edge", "moving_dark_edge", "moving_bright_bar", "moving_dark_bar", "moving_sinusoidal_grating", "contrast_reversed_edge", "moving_patch", "aperture_limited_edge"}:
        return float("nan")
    return float(int(result["preferred_index"]) == spec.direction_index)


def tensor_to_list(value: object) -> object:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


__all__ = [
    "ANGLES_DEG",
    "CONTRASTS",
    "DISPLACEMENTS",
    "STIMULUS_CLASSES",
    "StimulusSpec",
    "direction_accuracy",
    "evaluate_bank",
    "make_stimulus",
    "tensor_to_list",
]
