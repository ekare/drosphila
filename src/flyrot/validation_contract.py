"""Explicit validation contracts for image-plane direction and polarity audits."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import torch


CANONICAL_DIRECTION_NAMES = ("E", "SE", "S", "SW", "W", "NW", "N", "NE")
CANONICAL_DIRECTION_OFFSETS = ((1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1))
LEGACY_DIRECTION_NAMES = ("E", "NE", "N", "NW", "W", "SW", "S", "SE")
LEGACY_DIRECTION_OFFSETS = ((1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1))

# Physical bins are preserved; only the historical channel ordering changes.
LEGACY_TO_CANONICAL = (0, 7, 6, 5, 4, 3, 2, 1)
CANONICAL_TO_LEGACY = LEGACY_TO_CANONICAL


def _as_tensor_index(index: Iterable[int], value: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(tuple(index), dtype=torch.long, device=value.device)


def reorder_direction_vector(value: torch.Tensor, *, source: str = "legacy") -> torch.Tensor:
    """Reorder a tensor whose last dimension contains eight direction bins."""

    if value.shape[-1] != 8:
        raise ValueError(f"expected eight direction bins in the last dimension, got {tuple(value.shape)}")
    if source == "canonical":
        return value
    if source != "legacy":
        raise ValueError(f"unknown direction source order: {source!r}")
    return value.index_select(-1, _as_tensor_index(CANONICAL_TO_LEGACY, value))


def reorder_direction_channels(value: torch.Tensor, scales: tuple[int, ...], *, channel_dim: int, source: str = "legacy") -> torch.Tensor:
    """Reorder scale * direction channels while preserving scale order."""

    if source == "canonical":
        return value
    if source != "legacy":
        raise ValueError(f"unknown direction source order: {source!r}")
    channel_dim = channel_dim if channel_dim >= 0 else value.ndim + channel_dim
    expected = len(scales) * 8
    if value.shape[channel_dim] != expected:
        raise ValueError(f"expected {expected} direction channels, got {value.shape[channel_dim]}")
    moved = value.movedim(channel_dim, -1)
    shape = moved.shape
    grouped = moved.reshape(*shape[:-1], len(scales), 8)
    grouped = grouped.index_select(-1, _as_tensor_index(CANONICAL_TO_LEGACY, grouped))
    return grouped.reshape(*shape).movedim(-1, channel_dim)


def direction_vectors(*, order: str = "canonical", dtype: torch.dtype = torch.float32, device: torch.device | None = None) -> torch.Tensor:
    """Return normalized image-plane direction vectors in the requested order."""

    offsets = CANONICAL_DIRECTION_OFFSETS if order == "canonical" else LEGACY_DIRECTION_OFFSETS if order == "legacy" else None
    if offsets is None:
        raise ValueError(f"unknown direction order: {order!r}")
    vectors = torch.tensor(offsets, dtype=dtype, device=device)
    return vectors / torch.linalg.vector_norm(vectors, dim=-1, keepdim=True)


def direction_index(angle_deg: float) -> int:
    """Map an east-start clockwise image-plane angle to a canonical bin."""

    return int(round(float(angle_deg) / 45.0)) % 8


def signed_angle_delta_deg(predicted_deg: torch.Tensor | np.ndarray | float, target_deg: torch.Tensor | np.ndarray | float) -> torch.Tensor:
    """Return signed shortest predicted - target angles in [-180, 180)."""

    predicted = torch.as_tensor(predicted_deg, dtype=torch.float64)
    target = torch.as_tensor(target_deg, dtype=torch.float64)
    return (predicted - target + 180.0).remainder(360.0) - 180.0


def population_decode(response: torch.Tensor, *, tie_tolerance: float = 1e-6) -> dict[str, torch.Tensor]:
    """Decode canonical direction responses by argmax and vector population."""

    if response.shape[-1] != 8:
        raise ValueError(f"expected response[...,8], got {tuple(response.shape)}")
    vectors = direction_vectors(dtype=response.dtype, device=response.device)
    maximum = response.max(dim=-1, keepdim=True).values
    ties = (maximum - response).abs() <= float(tie_tolerance)
    argmax = response.argmax(dim=-1)
    vector = torch.einsum("...d,dc->...c", response, vectors)
    norm = torch.linalg.vector_norm(vector, dim=-1)
    angle = torch.rad2deg(torch.atan2(vector[..., 1], vector[..., 0])).remainder(360.0)
    population_index = torch.remainder(torch.round(angle / 45.0).long(), 8)
    return {
        "argmax_index": argmax,
        "tie_mask": ties,
        "tie_count": ties.sum(dim=-1),
        "population_vector": vector,
        "population_norm": norm,
        "population_angle_deg": angle,
        "population_index": population_index,
        "population_valid": norm > 1e-8,
    }


def response_metrics(response: torch.Tensor, expected_index: torch.Tensor, *, tie_tolerance: float = 1e-6) -> dict[str, float | int | None]:
    """Compute argmax, tie-aware, angular, cosine, and neighbor metrics."""

    if response.shape[-1] != 8:
        raise ValueError("response must end in eight bins")
    response = response.reshape(-1, 8)
    expected_index = torch.as_tensor(expected_index, device=response.device, dtype=torch.long).reshape(-1)
    if response.shape[0] != expected_index.shape[0]:
        raise ValueError("response and expected_index have different sample counts")
    decoded = population_decode(response, tie_tolerance=tie_tolerance)
    expected = expected_index
    expected_vectors = direction_vectors(device=response.device, dtype=response.dtype).index_select(0, expected)
    population_vector = decoded["population_vector"]
    population_norm = decoded["population_norm"]
    cosine = (population_vector * expected_vectors).sum(-1) / population_norm.clamp_min(1e-8)
    expected_angle = expected.to(response.dtype) * 45.0
    signed = signed_angle_delta_deg(decoded["population_angle_deg"], expected_angle).to(response.device)
    population_exact = decoded["population_index"].eq(expected)
    tie_aware = decoded["tie_mask"].gather(-1, expected[:, None]).squeeze(-1)
    difference = (decoded["argmax_index"] - expected).abs()
    circular_distance = torch.minimum(difference, 8 - difference)
    finite = torch.isfinite(signed) & decoded["population_valid"]

    def mean_bool(value: torch.Tensor) -> float | None:
        return float(value.float().mean()) if value.numel() else None

    def median(value: torch.Tensor) -> float | None:
        return float(value.median()) if value.numel() else None

    valid_signed = signed[finite]
    valid_cosine = cosine[finite]
    return {
        "samples": int(response.shape[0]),
        "argmax_accuracy": mean_bool(decoded["argmax_index"].eq(expected)),
        "tie_aware_accuracy": mean_bool(tie_aware),
        "population_exact_accuracy": mean_bool(population_exact),
        "population_neighbor_accuracy": mean_bool(circular_distance <= 1),
        "population_vector_median_error_deg": median(valid_signed.abs()),
        "population_vector_median_signed_error_deg": median(valid_signed),
        "population_vector_median_cosine": median(valid_cosine),
        "population_vector_cosine_sign_accuracy": mean_bool(valid_cosine > 0),
        "adjacent_tie_fraction": mean_bool(decoded["tie_count"] > 1),
        "population_undefined_fraction": float((~decoded["population_valid"]).float().mean()) if response.shape[0] else None,
        "tie_tolerance": float(tie_tolerance),
    }


def d4_matrices() -> dict[str, tuple[tuple[int, int], tuple[int, int]]]:
    """Return all eight signed-permutation transforms in image coordinates."""

    return {
        "identity": ((1, 0), (0, 1)),
        "rot90_cw": ((0, -1), (1, 0)),
        "rot180": ((-1, 0), (0, -1)),
        "rot270_cw": ((0, 1), (-1, 0)),
        "flip_x": ((-1, 0), (0, 1)),
        "flip_y": ((1, 0), (0, -1)),
        "flip_xy": ((-1, 0), (0, -1)),
        "reflect_swap": ((0, -1), (-1, 0)),
    }


def transform_response(response: torch.Tensor, matrix: tuple[tuple[int, int], tuple[int, int]]) -> torch.Tensor:
    """Apply a D4 mapping to canonical response bins."""

    if response.shape[-1] != 8:
        raise ValueError("response must end in eight bins")
    source_by_target: list[int] = []
    offsets = CANONICAL_DIRECTION_OFFSETS
    for dx, dy in offsets:
        observed = (matrix[0][0] * dx + matrix[0][1] * dy, matrix[1][0] * dx + matrix[1][1] * dy)
        source_by_target.append(offsets.index(observed))
    return response.index_select(-1, _as_tensor_index(source_by_target, response))


def d4_audit(response: torch.Tensor, expected_index: torch.Tensor, *, tie_tolerance: float = 1e-6) -> dict[str, object]:
    """Evaluate all D4 channel mappings without changing model weights."""

    result: dict[str, object] = {}
    for name, matrix in d4_matrices().items():
        result[name] = {"matrix": matrix, "metrics": response_metrics(transform_response(response, matrix), expected_index, tie_tolerance=tie_tolerance)}
    valid = [(name, values["metrics"]["population_vector_median_error_deg"]) for name, values in result.items() if values["metrics"]["population_vector_median_error_deg"] is not None]
    best = min(valid, key=lambda item: item[1])
    return {"transforms": result, "best_physical_mapping": best[0], "best_population_median_error_deg": best[1]}


def photoreceptor_variants(frames: torch.Tensor, *, local_window: int = 5) -> dict[str, dict[str, torch.Tensor]]:
    """Return at most three fixed, unlearned photoreceptor variants."""

    from flyrot.models.photoreceptor import Photoreceptor

    if frames.ndim == 4:
        frames = frames.unsqueeze(0)
    if frames.ndim != 5 or frames.shape[2] != 1:
        raise ValueError("frames must have shape T,1,H,W or B,T,1,H,W")
    log_gray = frames.clamp_min(0).add(1e-4).log()
    p0_on, p0_off = Photoreceptor(local_window=local_window)(frames)
    delta = torch.zeros_like(log_gray)
    delta[:, 1:] = log_gray[:, 1:] - log_gray[:, :-1]
    kernel = (local_window, local_window)
    pad = local_window // 2
    flat_delta = delta.reshape(-1, 1, delta.shape[-2], delta.shape[-1])
    common = torch.nn.functional.avg_pool2d(flat_delta, kernel, stride=1, padding=pad).reshape_as(delta)
    p1 = delta - common
    magnitude = torch.nn.functional.avg_pool2d(flat_delta.abs(), kernel, stride=1, padding=pad).reshape_as(delta)
    p2 = delta / (magnitude + 1e-3)
    return {
        "P0": {"pre": log_gray, "adapted": log_gray, "on": p0_on, "off": p0_off},
        "P1": {"pre": delta, "adapted": p1, "on": p1.clamp_min(0), "off": (-p1).clamp_min(0)},
        "P2": {"pre": delta, "adapted": p2, "on": p2.clamp_min(0), "off": (-p2).clamp_min(0)},
    }


def pure_edge_stimulus(
    kind: str,
    *,
    angle_deg: float,
    displacement: float,
    height: int = 128,
    width: int = 160,
    steps: int = 5,
    background: float = 0.1,
    foreground: float = 0.85,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create a one-transition ON or OFF half-plane edge.

    A pixel crosses the boundary at most once during the sequence. Bars,
    patches, gratings, and contrast reversals intentionally do not use this
    helper because they contain both leading and trailing/polarity events.
    """

    if kind not in {"on", "off"}:
        raise ValueError("pure edge kind must be 'on' or 'off'")
    radians = math.radians(float(angle_deg))
    direction = torch.tensor((math.cos(radians), math.sin(radians)), dtype=torch.float32)
    y = torch.arange(height, dtype=torch.float32)
    x = torch.arange(width, dtype=torch.float32)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    projection = xx * direction[0] + yy * direction[1]
    center_projection = ((width - 1) / 2.0) * direction[0] + ((height - 1) / 2.0) * direction[1]
    start = center_projection - displacement * (steps - 1) / 2.0
    frames = []
    for time_index in range(steps):
        threshold = start + displacement * time_index
        bright = projection <= threshold
        if kind == "off":
            bright = ~bright
        frames.append(torch.where(bright, torch.tensor(foreground), torch.tensor(background)))
    frames_tensor = torch.stack(frames)[:, None]
    margin = int(math.ceil(abs(displacement) * (steps - 1))) + 4
    valid = torch.zeros((1, height, width), dtype=torch.float32)
    if 2 * margin < min(height, width):
        valid[:, margin : height - margin, margin : width - margin] = 1.0
    return frames_tensor, valid


__all__ = [
    "CANONICAL_DIRECTION_NAMES",
    "CANONICAL_DIRECTION_OFFSETS",
    "CANONICAL_TO_LEGACY",
    "LEGACY_DIRECTION_NAMES",
    "LEGACY_DIRECTION_OFFSETS",
    "LEGACY_TO_CANONICAL",
    "d4_audit",
    "d4_matrices",
    "direction_index",
    "direction_vectors",
    "photoreceptor_variants",
    "population_decode",
    "pure_edge_stimulus",
    "reorder_direction_channels",
    "reorder_direction_vector",
    "response_metrics",
    "signed_angle_delta_deg",
    "transform_response",
]
