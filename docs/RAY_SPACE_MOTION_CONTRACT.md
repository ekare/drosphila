# Ray-space motion contract

The physical motion scale is angular or camera-ray displacement, not a fixed
pixel count. A raster-specific pixel displacement may be derived from focal
length for compatibility:

pixels = focal_pixels * tan(angle)
angle  = atan(pixels / focal_pixels)

Every raster and intrinsics configuration must retain its own calibration.
The ray grid is [x, y, 1] in the optical image frame and can optionally
invert Brown-Conrady distortion with a bounded iteration.

The current direction-cell bank still contains historical integer pixel
offsets and is therefore not an accepted ray-space production detector. The
phone golden benchmark validates the metadata and conversion contract only;
the real-texture field gate remains failed.

No uncalibrated response magnitude is reported as physical pixels, speed, or
metric translation.
