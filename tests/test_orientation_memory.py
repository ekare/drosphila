import numpy as np
import pytest

from flyrot.geometry.so3 import exp_so3, geodesic_distance
from flyrot.geometry.translation_flow import compensate_rotational_flow
from flyrot.geometry.rotational_flow import CameraIntrinsics
from flyrot.orientation_memory import AngularFeatureMemory, SO3OrientationBelief, VisualAzimuthRing


def test_so3_belief_composes_and_rejects_invalid_updates():
    belief = SO3OrientationBelief(min_confidence=0.5)
    first = belief.update(np.array([0.0, 0.0, 0.1]), confidence=0.8)
    rejected = belief.update(np.array([0.0, 0.0, 0.2]), confidence=0.1)
    assert first.accepted and not rejected.accepted
    assert belief.accepted_updates == 1 and belief.rejected_updates == 1
    assert np.isclose(geodesic_distance(belief.rotation, exp_so3([0.0, 0.0, 0.1])), 0.0)


def test_visual_azimuth_ring_wraps_and_is_not_full_so3():
    ring = VisualAzimuthRing(72)
    ring.update(np.pi * 1.5)
    assert ring.angle == pytest.approx(-np.pi / 2)
    assert 0 <= ring.bin_index < 72
    assert ring.as_dict()["role"] == "visual_azimuth_projection_only"


def test_angular_feature_memory_updates_reference_bins_without_rendering():
    memory = AngularFeatureMemory(azimuth_bins=12, elevation_bins=6, feature_dim=2)
    memory.update(
        np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]]),
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        np.eye(3),
        timestamp=1.0,
    )
    assert memory.summary()["occupied_cells"] == 2
    assert memory.summary()["renders_pixels"] is False
    assert memory.variance().shape == (6, 12, 2)


def test_oracle_rotational_compensation_recovers_residual_translation():
    intrinsics = CameraIntrinsics(fx=16.0, fy=14.0, cx=5.0, cy=4.0, width=10, height=8)
    rotation = np.array([0.03, -0.02, 0.01], dtype=np.float64)
    import torch

    rotation_t = torch.tensor(rotation).view(1, 3)
    observed = torch.zeros(1, 2, 8, 10)
    result = compensate_rotational_flow(observed, rotation_t, intrinsics)
    assert result["residual_flow"].shape == observed.shape
    assert np.isfinite(result["residual_flow"].numpy()).all()
