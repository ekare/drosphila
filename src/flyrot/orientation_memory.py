"""Oracle-safe orientation and latent angular feature memory primitives."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry.so3 import exp_so3


def _wrap_angle(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


@dataclass(frozen=True)
class OrientationUpdate:
    accepted: bool
    reason: str
    rotation: np.ndarray
    covariance: np.ndarray


class SO3OrientationBelief:
    """Stateful relative orientation composition with bounded rejection."""

    def __init__(self, *, reference_id: str = "initial", min_confidence: float = 0.0) -> None:
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence must be in [0, 1]")
        self.reference_id = reference_id
        self.min_confidence = float(min_confidence)
        self.rotation = np.eye(3, dtype=np.float64)
        self.covariance = np.zeros((3, 3), dtype=np.float64)
        self.accepted_updates = 0
        self.rejected_updates = 0

    def reset(self, *, reference_id: str | None = None) -> None:
        self.rotation = np.eye(3, dtype=np.float64)
        self.covariance = np.zeros((3, 3), dtype=np.float64)
        self.accepted_updates = 0
        self.rejected_updates = 0
        if reference_id is not None:
            self.reference_id = reference_id

    def update(
        self,
        rotation_vector: np.ndarray,
        *,
        confidence: float = 1.0,
        valid: bool = True,
        covariance: np.ndarray | None = None,
    ) -> OrientationUpdate:
        vector = np.asarray(rotation_vector, dtype=np.float64)
        if vector.shape != (3,) or not np.isfinite(vector).all():
            raise ValueError("rotation_vector must be finite shape (3,)")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        if not valid:
            self.rejected_updates += 1
            return OrientationUpdate(False, "invalid_measurement", self.rotation.copy(), self.covariance.copy())
        if confidence < self.min_confidence:
            self.rejected_updates += 1
            return OrientationUpdate(False, "low_confidence", self.rotation.copy(), self.covariance.copy())
        if covariance is not None:
            covariance = np.asarray(covariance, dtype=np.float64)
            if covariance.shape != (3, 3) or not np.isfinite(covariance).all():
                raise ValueError("covariance must be finite shape (3,3)")
            self.covariance = self.covariance + covariance / max(confidence, 1e-6)
        self.rotation = self.rotation @ exp_so3(vector)
        self.accepted_updates += 1
        return OrientationUpdate(True, "accepted", self.rotation.copy(), self.covariance.copy())

    def as_dict(self) -> dict[str, object]:
        return {
            "reference_id": self.reference_id,
            "rotation": self.rotation.tolist(),
            "covariance": self.covariance.tolist(),
            "accepted_updates": self.accepted_updates,
            "rejected_updates": self.rejected_updates,
        }


class VisualAzimuthRing:
    """A 1-D visual azimuth projection, never a replacement for SO(3)."""

    def __init__(self, bins: int = 72) -> None:
        if bins < 8:
            raise ValueError("bins must be at least 8")
        self.bins = int(bins)
        self.angle = 0.0

    def update(self, delta_yaw_rad: float, *, valid: bool = True) -> float:
        if valid:
            self.angle = _wrap_angle(self.angle + float(delta_yaw_rad))
        return self.angle

    @property
    def bin_index(self) -> int:
        return int(np.floor((self.angle + np.pi) / (2.0 * np.pi) * self.bins)) % self.bins

    def as_dict(self) -> dict[str, object]:
        return {"bins": self.bins, "angle_rad": self.angle, "bin_index": self.bin_index, "role": "visual_azimuth_projection_only"}


class AngularFeatureMemory:
    """Compact orientation-aligned feature memory; it does not render pixels."""

    def __init__(self, *, azimuth_bins: int = 72, elevation_bins: int = 18, feature_dim: int = 8) -> None:
        if min(azimuth_bins, elevation_bins, feature_dim) < 1:
            raise ValueError("memory dimensions must be positive")
        self.azimuth_bins = int(azimuth_bins)
        self.elevation_bins = int(elevation_bins)
        self.feature_dim = int(feature_dim)
        shape = (self.elevation_bins, self.azimuth_bins, self.feature_dim)
        self.mean = np.zeros(shape, dtype=np.float64)
        self.second_moment = np.zeros(shape, dtype=np.float64)
        self.confidence = np.zeros(shape[:2], dtype=np.float64)
        self.count = np.zeros(shape[:2], dtype=np.int64)
        self.last_timestamp = np.full(shape[:2], np.nan, dtype=np.float64)

    def _indices(self, rays: np.ndarray, orientation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rays = np.asarray(rays, dtype=np.float64)
        orientation = np.asarray(orientation, dtype=np.float64)
        if rays.ndim != 2 or rays.shape[1] != 3:
            raise ValueError("rays must have shape N,3")
        if orientation.shape != (3, 3):
            raise ValueError("orientation must have shape 3,3")
        reference_rays = (orientation @ rays.T).T
        azimuth = np.arctan2(reference_rays[:, 1], reference_rays[:, 0])
        elevation = np.arctan2(reference_rays[:, 2], np.linalg.norm(reference_rays[:, :2], axis=1))
        azimuth_index = np.floor((azimuth + np.pi) / (2.0 * np.pi) * self.azimuth_bins).astype(np.int64) % self.azimuth_bins
        elevation_index = np.floor((elevation + np.pi / 2.0) / np.pi * self.elevation_bins).astype(np.int64)
        return azimuth_index, np.clip(elevation_index, 0, self.elevation_bins - 1)

    def update(
        self,
        rays: np.ndarray,
        features: np.ndarray,
        orientation: np.ndarray,
        *,
        confidence: np.ndarray | float = 1.0,
        timestamp: float | None = None,
    ) -> None:
        features = np.asarray(features, dtype=np.float64)
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError("features must have shape N,feature_dim")
        if features.shape[0] != len(rays):
            raise ValueError("rays and features must have the same N")
        confidence = np.broadcast_to(np.asarray(confidence, dtype=np.float64), (features.shape[0],))
        if np.any(confidence < 0) or not np.isfinite(features).all():
            raise ValueError("confidence must be non-negative and features finite")
        azimuth_index, elevation_index = self._indices(rays, orientation)
        for index in range(features.shape[0]):
            e, a = elevation_index[index], azimuth_index[index]
            weight = float(confidence[index])
            if weight <= 0:
                continue
            old_count = float(self.count[e, a])
            new_count = old_count + weight
            delta = features[index] - self.mean[e, a]
            self.mean[e, a] += weight * delta / new_count
            self.second_moment[e, a] += weight * (features[index] * features[index] - self.second_moment[e, a]) / new_count
            self.count[e, a] = int(round(new_count))
            self.confidence[e, a] = min(1.0, self.confidence[e, a] + weight)
            if timestamp is not None:
                self.last_timestamp[e, a] = float(timestamp)

    def variance(self) -> np.ndarray:
        return np.maximum(self.second_moment - np.square(self.mean), 0.0)

    def summary(self) -> dict[str, object]:
        return {
            "azimuth_bins": self.azimuth_bins,
            "elevation_bins": self.elevation_bins,
            "feature_dim": self.feature_dim,
            "occupied_cells": int(np.count_nonzero(self.count)),
            "total_observations": int(self.count.sum()),
            "size_bytes": int(self.mean.nbytes + self.second_moment.nbytes + self.confidence.nbytes + self.count.nbytes + self.last_timestamp.nbytes),
            "renders_pixels": False,
            "representation": "latent angular feature memory, not a 3D world map",
        }


__all__ = ["AngularFeatureMemory", "OrientationUpdate", "SO3OrientationBelief", "VisualAzimuthRing"]
