# Real-texture pure-rotation dataset

The benchmark is **real-texture controlled geometry**, not synthetic texture.
RGB frames are read online from trajectory-disjoint TartanAir splits and
warped with `H = K_target @ R_image @ inverse(K_source)`. No source dataset
file is copied or modified.

- Generator commit: `abfa1118abec8d586da2b61f5a14f33a5f7f4031`
- Source split manifest SHA256: `808f6cc775c986f3ba8b88180e0bf7995895b37db817c747bed7b1baf10b9c94`
- Per split: 48 deterministic examples for train, validation, and test
- Windows: T=3, T=5, T=7
- Raster: 128x128 plus the non-square diagnostics elsewhere in the project
- Families: zero, yaw, pitch, roll, combined two-axis, combined three-axis
- Magnitudes: 0, 0.25, 0.5, 1, 2, 5, 10, 15 degrees
- V1-A checkpoint SHA256: `3d26a94a50a30a5b95bc30f9e982e045b201c88b8d5d3e8e4e2409bb49bcf81f`

Each example carries a path-free logical image ID, intrinsics, per-step and
endpoint SO(3), dense rotational flow, valid warp mask, resize transform,
texture observability statistics, and seed. Test was evaluated only as a
post-selection diagnostic; it was not used for selection.
