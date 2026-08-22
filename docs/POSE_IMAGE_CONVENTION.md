# Pose and image-motion convention

FlyRot keeps the pose-to-image mapping explicit because a sign or transpose
mistake can look like a plausible learning result.

## Source pose

TartanAir pose rows are `tx ty tz qx qy qz qw`. The rotation is interpreted as
`R_world_camera` in the TartanAir NED frame: x=forward, y=right, z=down.
For consecutive poses, the camera-relative rotation is:

```text
R_camera_relative = R_t.T @ R_t+1
```

The image-motion target reverses that active mapping:

```text
R_image_motion = R_camera_relative.T
```

The optical camera frame is x=right, y=down, z=forward. The fixed basis
change is `NED_TO_OPTICAL @ R @ NED_TO_OPTICAL.T`.

## Pixels and intrinsics

`CameraIntrinsics` stores `fx, fy, cx, cy` in pixels. Normalized rays are
`x=(u-cx)/fx`, `y=(v-cy)/fy`, with ray `[x,y,1]`. A crop subtracts its left/top
offset from the principal point. A resize uses pixel-center mapping:

```text
p_out = (p_crop + 0.5) * scale - 0.5
```

The released v0.3 baseline uses the historical centered API for compatibility.
The TartanAir V2 `f=320, (cx,cy)=(320,320)` manifest is tracked separately as
`assumed` until the exact camera calibration is verified.

## Validation requirement

Any new model or report must pass exact homography golden cases for non-square
and off-center intrinsics, and must state whether it uses camera-relative or
image-motion rotation. A correct-sign and wrong-sign case are required; an
axis label alone is not acceptance evidence.
