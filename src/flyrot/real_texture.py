"""Online deterministic real-texture pure-rotation benchmark primitives."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from flyrot.data.tartanair import default_split
from flyrot.geometry.camera import tartanair_v2_lcam_front_intrinsics
from flyrot.geometry.so3 import exp_so3, log_so3
from flyrot.geometry.rotational_flow import CameraIntrinsics, rotational_flow_basis
from flyrot.validation_contract import direction_vectors, reorder_direction_vector


ROTATION_FAMILIES = ("zero", "yaw", "pitch", "roll", "two_axis", "three_axis")
ROTATION_MAGNITUDES_DEG = (0.0, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0)


def frame_key(path: Path) -> int:
    return int(path.name.split("_", 1)[0])


def path_free_record(record: dict) -> dict:
    return {key: record[key] for key in ("environment", "difficulty", "trajectory", "camera", "rgb_count", "resolution", "pose_format") if key in record}


def manifest_sha256(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def rotation_vector(family: str, magnitude_deg: float, sign: int) -> np.ndarray:
    angle = math.radians(float(magnitude_deg)) * float(sign)
    if family == "zero":
        return np.zeros(3, dtype=np.float64)
    if family == "yaw":
        return np.asarray([0.0, angle, 0.0])
    if family == "pitch":
        return np.asarray([angle, 0.0, 0.0])
    if family == "roll":
        return np.asarray([0.0, 0.0, angle])
    if family == "two_axis":
        return np.asarray([angle, 0.65 * angle, 0.0])
    if family == "three_axis":
        return np.asarray([angle, 0.65 * angle, 0.4 * angle])
    raise ValueError(f"unknown rotation family {family!r}")


def exact_homography(rotation: np.ndarray, intrinsics: CameraIntrinsics) -> np.ndarray:
    matrix = np.asarray(rotation, dtype=np.float64)
    camera = np.asarray([[intrinsics.fx, 0.0, intrinsics.cx], [0.0, intrinsics.fy, intrinsics.cy], [0.0, 0.0, 1.0]])
    return camera @ matrix @ np.linalg.inv(camera)


def dense_homography_flow(rotation: np.ndarray, intrinsics: CameraIntrinsics) -> tuple[np.ndarray, np.ndarray]:
    height, width = intrinsics.height, intrinsics.width
    y, x = np.meshgrid(np.arange(height, dtype=np.float64), np.arange(width, dtype=np.float64), indexing="ij")
    points = np.stack((x, y, np.ones_like(x)), axis=-1)
    projected = points @ exact_homography(rotation, intrinsics).T
    projected = projected[..., :2] / np.maximum(projected[..., 2:3], 1e-8)
    flow = projected - points[..., :2]
    valid = (
        np.isfinite(projected).all(axis=-1)
        & (projected[..., 0] >= 0)
        & (projected[..., 0] < width)
        & (projected[..., 1] >= 0)
        & (projected[..., 1] < height)
    )
    return flow.astype(np.float32), valid.astype(np.float32)


def read_rgb(path: Path, size: tuple[int, int]) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    width, height = size
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0


def _warp(image: np.ndarray, rotation: np.ndarray, intrinsics: CameraIntrinsics) -> tuple[np.ndarray, np.ndarray]:
    homography = exact_homography(rotation, intrinsics)
    height, width = intrinsics.height, intrinsics.width
    warped = cv2.warpPerspective(image, homography, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    source_mask = np.ones((height, width), dtype=np.float32)
    valid = cv2.warpPerspective(source_mask, homography, (width, height), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    return warped, (valid > 0.5).astype(np.float32)


@dataclass(frozen=True)
class RotationExample:
    source_logical_id: str
    source_path: str
    family: str
    magnitude_deg: float
    sign: int
    window_length: int
    seed: int


class RealTextureRotationDataset(Dataset):
    """Trajectory-disjoint, online exact-homography texture dataset.

    Absolute source paths are runtime inputs only and never appear in the
    path-free manifest returned by :meth:`manifest`.
    """

    def __init__(
        self,
        records: list[dict],
        *,
        window_length: int = 3,
        image_size: tuple[int, int] = (128, 128),
        limit: int = 48,
        seed: int = 20260823,
        horizontal_fov_deg: float | None = None,
    ) -> None:
        if window_length not in {3, 5, 7, 9, 17}:
            raise ValueError("window_length must be one of 3, 5, 7, 9, or 17")
        self.records = records
        self.window_length = window_length
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.seed = int(seed)
        if horizontal_fov_deg is not None and not 1.0 < float(horizontal_fov_deg) < 179.0:
            raise ValueError("horizontal_fov_deg must be between 1 and 179 degrees")
        self.horizontal_fov_deg = None if horizontal_fov_deg is None else float(horizontal_fov_deg)
        self.examples: list[RotationExample] = []
        families = ("zero", "yaw", "pitch", "roll", "two_axis", "three_axis")
        for index in range(max(0, int(limit))):
            record = records[index % len(records)]
            paths = sorted(Path(record["rgb_root"]).glob("*.png"), key=frame_key)
            if not paths:
                raise FileNotFoundError(record["rgb_root"])
            source_index = (index * 37 + seed) % len(paths)
            family = families[index % len(families)]
            magnitude = ROTATION_MAGNITUDES_DEG[(index // len(families)) % len(ROTATION_MAGNITUDES_DEG)]
            sign = 1 if ((index // len(families)) % 2 == 0) else -1
            logical_id = f"{record['environment']}/{record['difficulty']}/{record['trajectory']}/{record['camera']}/{frame_key(paths[source_index]):06d}"
            self.examples.append(RotationExample(logical_id, str(paths[source_index]), family, magnitude, sign, window_length, seed + index))
        self._cache: dict[int, dict[str, object]] = {}

    def __len__(self) -> int:
        return len(self.examples)

    def manifest(self) -> dict:
        return {
            "kind": "real-texture-controlled-geometry",
            "window_length": self.window_length,
            "image_size": list(self.image_size),
            "seed": self.seed,
            "horizontal_fov_deg": self.horizontal_fov_deg,
            "examples": [
                {
                    "source_logical_id": item.source_logical_id,
                    "family": item.family,
                    "magnitude_deg": item.magnitude_deg,
                    "sign": item.sign,
                    "window_length": item.window_length,
                    "seed": item.seed,
                }
                for item in self.examples
            ],
        }

    def __getitem__(self, index: int) -> dict[str, object]:
        if index in self._cache:
            return self._cache[index]
        item = self.examples[index]
        width, height = self.image_size
        intrinsics = tartanair_v2_lcam_front_intrinsics(640, 640).resized(width, height)
        if self.horizontal_fov_deg is not None:
            focal_x = (width / 2.0) / math.tan(math.radians(self.horizontal_fov_deg) / 2.0)
            focal_y = focal_x * height / width
            intrinsics = CameraIntrinsics(
                fx=focal_x,
                fy=focal_y,
                cx=intrinsics.cx,
                cy=intrinsics.cy,
                width=width,
                height=height,
            )
        base = read_rgb(Path(item.source_path), self.image_size)
        endpoint = rotation_vector(item.family, item.magnitude_deg, item.sign)
        step_vector = endpoint / float(item.window_length - 1)
        step_rotation = exp_so3(step_vector)
        cumulative = np.eye(3, dtype=np.float64)
        frames = [base]
        flows = []
        masks = []
        step_vectors = []
        for _ in range(item.window_length - 1):
            cumulative = cumulative @ step_rotation
            frame, mask = _warp(base, cumulative, intrinsics)
            frames.append(frame)
            flow, flow_mask = dense_homography_flow(step_rotation, intrinsics)
            flows.append(flow)
            masks.append(mask * flow_mask)
            step_vectors.append(log_so3(step_rotation[None])[0].astype(np.float32))
        gray = cv2.cvtColor((base * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        texture = np.sqrt(gx * gx + gy * gy)
        texture_threshold = float(np.quantile(texture, 0.35))
        observability = (texture >= texture_threshold).astype(np.float32)
        result = {
            "frames": torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).contiguous(),
            "target_rotation_vector": torch.from_numpy(endpoint.astype(np.float32)),
            "target_step_rotation_vectors": torch.from_numpy(np.stack(step_vectors)),
            "dense_flow": torch.from_numpy(np.stack(flows)).permute(0, 3, 1, 2).contiguous(),
            "valid_mask": torch.from_numpy(np.stack(masks)),
            "observability_mask": torch.from_numpy(np.stack(masks)) * torch.from_numpy(observability[None]),
            "texture_statistics": {"gradient_median": float(np.median(texture)), "gradient_p90": float(np.quantile(texture, 0.9)), "gradient_threshold": texture_threshold},
            "metadata": {"source_logical_id": item.source_logical_id, "family": item.family, "magnitude_deg": item.magnitude_deg, "sign": item.sign, "window_length": item.window_length, "seed": item.seed, "horizontal_fov_deg": self.horizontal_fov_deg, "intrinsics": intrinsics.as_dict(), "crop_resize": {"source": [640, 640], "output": [width, height], "operation": "resize_only"}},
        }
        self._cache[index] = result
        return result


def decode_direction_population(energy: torch.Tensor, scales: tuple[int, ...], field_size: int = 8, *, direction_order: str = "legacy") -> dict[str, torch.Tensor]:
    """Decode ON/OFF populations with legacy compatibility or canonical audit order."""

    if energy.ndim != 5 or energy.shape[2] != 2 * len(scales) * 8:
        raise ValueError("expected concatenated ON/OFF energy")
    batch, steps, _, height, width = energy.shape
    by = energy.reshape(batch, steps, 2, len(scales), 8, height, width)
    pooled = torch.nn.functional.adaptive_avg_pool2d(by.reshape(batch * steps * 2 * len(scales) * 8, 1, height, width), (field_size, field_size))
    pooled = pooled.reshape(batch, steps, 2, len(scales), 8, field_size, field_size)
    polarity_direction = pooled.sum(dim=3)
    if direction_order == "canonical":
        polarity_direction = reorder_direction_vector(polarity_direction.movedim(3, -1), source="legacy").movedim(-1, 3)
        vectors = direction_vectors(order="canonical", dtype=energy.dtype, device=energy.device)
    elif direction_order == "legacy":
        vectors = direction_vectors(order="legacy", dtype=energy.dtype, device=energy.device)
    else:
        raise ValueError(f"unknown direction_order: {direction_order!r}")
    direction_vector = torch.einsum("btqdhw,dc->btqchw", polarity_direction, vectors)
    scale_values = torch.tensor(scales, dtype=energy.dtype, device=energy.device)
    scale_strength = pooled.sum(dim=4)
    magnitude = (scale_strength * scale_values.view(1, 1, 1, -1, 1, 1)).sum(dim=3) / scale_strength.sum(dim=3).clamp_min(1e-8)
    confidence = torch.linalg.vector_norm(direction_vector, dim=3)
    unit = direction_vector / confidence.unsqueeze(3).clamp_min(1e-8)
    velocity = unit * magnitude.unsqueeze(3)
    return {"polarity_direction": polarity_direction, "direction_vector": direction_vector, "confidence": confidence, "magnitude": magnitude, "velocity": velocity, "direction_order": direction_order}


def solve_weighted_rotation(field: torch.Tensor, weights: torch.Tensor, intrinsics: CameraIntrinsics, *, iterations: int = 3) -> dict[str, torch.Tensor]:
    """Solve ``flow = pixel_rotational_basis * omega`` with bounded IRLS."""

    if field.ndim != 5 or field.shape[2] != 2:
        raise ValueError("field must have B,T,2,H,W")
    x = (torch.arange(intrinsics.width, device=field.device, dtype=field.dtype) - intrinsics.cx) / intrinsics.fx
    y = (torch.arange(intrinsics.height, device=field.device, dtype=field.dtype) - intrinsics.cy) / intrinsics.fy
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    basis = rotational_flow_basis(xx, yy).permute(2, 3, 1, 0).clone()
    basis[..., 0, :] *= intrinsics.fx
    basis[..., 1, :] *= intrinsics.fy
    b, t = field.shape[:2]
    matrix = basis.reshape(-1, 2, 3)
    observation = field.permute(0, 1, 3, 4, 2).reshape(b, t, -1, 2)
    base_weights = weights.reshape(b, t, -1).clamp_min(0)
    robust_weights = base_weights
    solution = torch.zeros((b, t, 3), device=field.device, dtype=field.dtype)
    for _ in range(iterations):
        normal = torch.einsum("pij,btp,pik->btjk", matrix, robust_weights, matrix)
        rhs = torch.einsum("pij,btp,btpi->btj", matrix, robust_weights, observation)
        solution = torch.linalg.solve(normal + 1e-4 * torch.eye(3, device=field.device, dtype=field.dtype).view(1, 1, 3, 3), rhs)
        residual = observation - torch.einsum("pij,btj->btpi", matrix, solution)
        norm = torch.linalg.vector_norm(residual, dim=-1)
        robust_weights = base_weights / (1.0 + norm / 2.0)
    residual = observation - torch.einsum("pij,btj->btpi", matrix, solution)
    normal = torch.einsum("pij,btp,pik->btjk", matrix, robust_weights, matrix)
    eigenvalues = torch.linalg.eigvalsh(normal + 1e-4 * torch.eye(3, device=field.device, dtype=field.dtype).view(1, 1, 3, 3))
    condition = eigenvalues[..., -1] / eigenvalues[..., 0].clamp_min(1e-8)
    return {"rotation_vector": solution, "residual": residual, "normal": normal, "condition": condition, "valid": torch.isfinite(solution).all(dim=-1) & (condition < 1e8)}


__all__ = [
    "ROTATION_FAMILIES",
    "ROTATION_MAGNITUDES_DEG",
    "RealTextureRotationDataset",
    "decode_direction_population",
    "dense_homography_flow",
    "manifest_sha256",
    "path_free_record",
    "rotation_vector",
    "solve_weighted_rotation",
]
