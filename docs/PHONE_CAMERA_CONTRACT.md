# Phone camera contract

Production input is one normal perspective RGB camera:

frames: B,T,3,H,W
intrinsics: fx, fy, cx, cy, width, height
timestamps: strictly increasing seconds
distortion: optional Brown-Conrady metadata
exposure: optional per-frame metadata
rolling_shutter_line_time: optional
crop_resize_history: optional
camera_id: optional

The contract supports arbitrary aspect ratios and does not assume square
images, centered principal points, equal focal lengths, fixed frame interval,
or panorama input. The reference phone rasters are 320x180, 256x144, and
192x108; 128x128 remains a compatibility baseline only.

Pixels use x-right, y-down, z-forward optical coordinates. Crop and resize
operations update intrinsics with pixel-center mapping. Distortion is explicit
metadata and is never silently inferred from a filename.

The implementation is in flyrot.geometry.phone_camera. It provides
CameraDistortion, FrameTimestamp, ExposureMetadata,
RollingShutterMetadata, PhoneCameraInput, CameraRayGrid, and validation
helpers. It is a geometry contract, not a claim that a mobile export exists.
