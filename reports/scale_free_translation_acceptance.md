# Scale-free translation acceptance

status: **oracle_only**

evidence: `rigid_camera_flow` decomposes depth-conditioned rigid flow into
rotation and translation residuals; `compensate_rotational_flow` subtracts an
exact or predicted rotational flow; `scale_free_translation_direction`
normalizes translation without inventing metric scale. Geometry tests pass.

blocker: No accepted local field and no usable indexed depth trajectory.

next_required_condition: Pass F0 and run pure-rotation, pure-translation,
mixed-motion, near-zero, lateral, vertical, and forward/backward gates on a
depth-verified split.
