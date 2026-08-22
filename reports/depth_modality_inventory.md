# Depth/flow modality inventory

## Indexed fast-cache

The 84 trajectory-camera records in the active fast-cache RGB+pose index have
`depth_count=0`, `flow_count=0`, and `segmentation_count=0`. Targeted sibling
directory checks on the indexed TartanAir trajectory roots found no matching
depth, flow, or segmentation asset.

## Other known roots

Targeted metadata inspection found 22 `data.pt` records under a separate
precomputed store on the data disk. A representative record contains
`features`, `teacher_depth`, `teacher_mask`, `poses`, and compressed JPEGs;
`teacher_depth` is 80x80, masked, and has values approximately 0 to 0.357,
while the JPEG raster is 320x240. This is a teacher prediction artifact, not a
native TartanAir depth manifest. Its metric type, scale, calibration, and
frame alignment are not verified, so it is not usable as a depth-backed gold
subset.

An unrelated TUM RGB-D depth directory also exists on the data disk. It is not
trajectory-aligned with the TartanAir RGB index and is not used here.

Conclusion: the real directional/depth oracle remains unresolved. No dataset
was copied, moved, or modified.
