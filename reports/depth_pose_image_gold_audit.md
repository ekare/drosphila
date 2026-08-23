# Depth-backed pose/image gold audit

Status: **not verified / blocked**.

The exact real-texture pure-rotation benchmark is valid without depth and uses
the documented TartanAir V2 calibration assumption after an explicit resize.
The fast-cache has no native depth or flow. The separate `teacher_depth`
artifacts lack provenance and metric-depth/type calibration and use a different
320x240 representation, so they cannot be promoted to a depth-backed oracle.

Consequently no rigid-flow, occlusion, target-depth-consistency, or
depth-backed pose/image acceptance claim is made. Translation and scene memory
remain outside this task.
