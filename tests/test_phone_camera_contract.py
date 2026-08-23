import numpy as np
import pytest
import torch

from flyrot.geometry.camera import CropResizeTransform
from flyrot.geometry.phone_camera import (
    CameraDistortion,
    ExposureMetadata,
    FrameTimestamp,
    PhoneCameraInput,
    RollingShutterMetadata,
    apply_crop_resize_history,
    camera_ray_grid,
    distort_normalized,
    undistort_normalized,
    validate_phone_frames,
)
from flyrot.geometry.rotational_flow import CameraIntrinsics


def test_arbitrary_aspect_off_center_ray_grid_uses_pixel_coordinates():
    intrinsics = CameraIntrinsics(fx=140.0, fy=96.0, cx=81.25, cy=43.75, width=192, height=108)
    grid = camera_ray_grid(intrinsics)
    assert grid.rays.shape == (108, 192, 3)
    assert torch.allclose(grid.rays[0, 0], torch.tensor([(-81.25) / 140.0, (-43.75) / 96.0, 1.0]))
    assert grid.valid.all()


def test_crop_resize_history_preserves_ray_geometry():
    source = CameraIntrinsics(fx=320.0, fy=320.0, cx=320.0, cy=320.0, width=640, height=640)
    history = (CropResizeTransform(left=0, top=140, crop_width=640, crop_height=360, output_width=320, output_height=180),)
    output = apply_crop_resize_history(source, history)
    assert output.width == 320 and output.height == 180
    assert output.fx == 160.0 and output.fy == 160.0
    assert output.cx == 159.75 and output.cy == 89.75


def test_brown_conrady_round_trip_and_zero_model():
    points = torch.tensor([[-0.4, -0.2], [0.0, 0.0], [0.3, 0.25]], dtype=torch.float64)
    distortion = CameraDistortion(model="brown_conrady", k1=0.04, k2=-0.01, p1=0.002, p2=-0.001)
    distorted = distort_normalized(points, distortion)
    recovered = undistort_normalized(distorted, distortion, iterations=12)
    assert torch.allclose(recovered, points, atol=1e-7, rtol=1e-7)
    assert torch.equal(distort_normalized(points, CameraDistortion()), points)


def test_phone_input_contract_rejects_bad_timing_or_raster():
    intrinsics = CameraIntrinsics.centered(320, 180)
    camera = PhoneCameraInput(
        intrinsics=intrinsics,
        timestamps=(FrameTimestamp(0.0), FrameTimestamp(0.02)),
        exposure=(ExposureMetadata(exposure_seconds=0.01), ExposureMetadata(exposure_seconds=0.01)),
        rolling_shutter=RollingShutterMetadata(line_time_seconds=1e-5),
    )
    frames = torch.zeros(1, 2, 3, 180, 320)
    validate_phone_frames(frames, camera)
    with pytest.raises(ValueError):
        validate_phone_frames(torch.zeros(1, 2, 3, 128, 128), camera)
    with pytest.raises(ValueError):
        PhoneCameraInput(intrinsics, (FrameTimestamp(0.1), FrameTimestamp(0.1))).validate_timestamps()


def test_distorted_ray_grid_is_finite_for_phone_rasters():
    intrinsics = CameraIntrinsics(fx=155.0, fy=148.0, cx=96.2, cy=53.8, width=192, height=108)
    grid = camera_ray_grid(intrinsics, CameraDistortion(model="brown_conrady", k1=0.03, k2=-0.01))
    assert grid.valid.all()
    assert np.isfinite(grid.rays.cpu().numpy()).all()
