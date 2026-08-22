"""Deterministic synthetic scenes for rotation-residual diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch

from .diagnostics import exact_rotational_flow
from .geometry.rotational_flow import _hat


@dataclass
class SyntheticCase:
    name: str
    frames: torch.Tensor
    pair_rotations: torch.Tensor
    oracle_energy: torch.Tensor
    oracle_on_energy: torch.Tensor
    oracle_off_energy: torch.Tensor


def _base_texture(height: int, width: int, seed: int) -> np.ndarray:
    generator = np.random.default_rng(seed)
    y = np.linspace(-1.0, 1.0, height, dtype=np.float32)[:, None]
    x = np.linspace(-1.0, 1.0, width, dtype=np.float32)[None, :]
    red = 0.5 + 0.22 * np.sin(7.0 * x + 3.0 * y) + 0.12 * np.cos(13.0 * y)
    green = 0.5 + 0.20 * np.cos(5.0 * x - 8.0 * y) + 0.10 * np.sin(17.0 * x)
    blue = 0.5 + 0.18 * np.sin(9.0 * (x + y)) + 0.12 * np.cos(11.0 * x)
    image = np.stack((red, green, blue), axis=-1)
    image += generator.normal(0.0, 0.015, image.shape).astype(np.float32)
    return np.clip(image, 0.0, 1.0)


def _rotation_matrix(rotation: torch.Tensor) -> np.ndarray:
    return torch.matrix_exp(_hat(rotation)).detach().cpu().numpy()


def _warp_rotation(image: np.ndarray, rotation: torch.Tensor) -> np.ndarray:
    height, width = image.shape[:2]
    focal_x = width / 2.0
    focal_y = height / 2.0
    camera = np.array(
        [[focal_x, 0.0, (width - 1) / 2.0], [0.0, focal_y, (height - 1) / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    homography = camera @ _rotation_matrix(rotation) @ np.linalg.inv(camera)
    return cv2.warpPerspective(
        image.astype(np.float32),
        homography,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )


def _warp_translation(image: np.ndarray, dx: float, dy: float) -> np.ndarray:
    height, width = image.shape[:2]
    matrix = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    return cv2.warpAffine(image.astype(np.float32), matrix, (width, height), borderMode=cv2.BORDER_REFLECT)


def _add_moving_patch(image: np.ndarray, step: int) -> np.ndarray:
    result = image.copy()
    height, width = image.shape[:2]
    # Keep the moving object spatially localized at every smoke-test
    # resolution; a fixed one-sixth square becomes too large in normalized
    # entropy support at 128x128.
    size = max(6, min(height, width) // 8)
    left = min(width - size, width // 4 + step * max(2, width // 32))
    top = height // 3
    result[top : top + size, left : left + size] = np.array([0.95, 0.15, 0.08], dtype=np.float32)
    return result


def _direction_energy(flow: torch.Tensor, scales: int = 3) -> torch.Tensor:
    """Convert a diagnostic vector field into quantized direction evidence."""

    if flow.ndim != 5 or flow.shape[2] != 2:
        raise ValueError("flow must have shape B,T,2,H,W")
    angles = torch.arange(8, dtype=flow.dtype, device=flow.device) * (2.0 * torch.pi / 8)
    directions = torch.stack((angles.cos(), angles.sin()), dim=1)
    unit = flow / torch.linalg.vector_norm(flow, dim=2, keepdim=True).clamp_min(1e-8)
    compatibility = torch.einsum("dp,btphw->btdhw", directions, unit).clamp_min(0)
    compatibility = compatibility * (torch.linalg.vector_norm(flow, dim=2, keepdim=True) > 1e-8)
    return compatibility.repeat(1, 1, scales, 1, 1)


def make_synthetic_case(name: str, height: int = 128, width: int = 128, seed: int = 0) -> SyntheticCase:
    """Create RGB frames and an analytic direction-energy oracle."""

    names = {"zero", "rotation", "brightness", "translation", "moving_patch", "mixed"}
    if name not in names:
        raise ValueError(f"unknown synthetic case {name!r}; choose from {sorted(names)}")
    base = _base_texture(height, width, seed)
    pair_rotation = torch.tensor([0.08, -0.06, 0.05], dtype=torch.float32)
    frames: list[np.ndarray] = []
    for step in range(3):
        if name in {"rotation", "mixed"}:
            image = _warp_rotation(base, pair_rotation * step)
        else:
            image = base.copy()
        if name in {"translation", "mixed"}:
            image = _warp_translation(image, 3.0 * step, -1.5 * step)
        if name in {"moving_patch", "mixed"}:
            image = _add_moving_patch(image, step)
        if name == "brightness":
            image = image * (0.75 + 0.25 * step)
        frames.append(np.clip(image, 0.0, 1.0))
    frame_tensor = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).contiguous()

    pair_rotations = pair_rotation.repeat(2, 1) if name in {"rotation", "mixed"} else torch.zeros(2, 3)
    flow = exact_rotational_flow(pair_rotations.unsqueeze(0), height, width)
    if name in {"translation", "mixed"}:
        yy = torch.linspace(-1.0, 1.0, height).view(1, 1, 1, height, 1)
        translation = torch.zeros_like(flow)
        translation[:, :, 0] = 0.04 + 0.02 * yy
        translation[:, :, 1] = -0.02
        flow = flow + translation
    if name == "moving_patch":
        flow = torch.zeros_like(flow)
        patch_size = max(6, min(height, width) // 8)
        flow[:, :, 0, height // 3 : height // 3 + patch_size, width // 4 : width // 4 + patch_size] = 0.12
    if name == "brightness":
        energy = torch.full((1, 2, 24, height, width), 0.15, dtype=torch.float32)
    else:
        energy = _direction_energy(flow)
    if name == "moving_patch":
        energy = energy * 1.5
    if name == "mixed":
        energy = energy + 0.25 * _direction_energy(torch.roll(flow, shifts=(6, -4), dims=(-2, -1)))
    on_energy = energy * (0.6 if name != "brightness" else 1.0)
    off_energy = energy * (0.4 if name != "brightness" else 0.0)
    return SyntheticCase(name, frame_tensor, pair_rotations, energy, on_energy, off_energy)
