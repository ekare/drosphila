# Depth and flow modality inventory

status: **blocked** for learned translation.

evidence: The active RGB+pose index has zero indexed depth records and zero
indexed flow records in the existing modality inventory. The repository's
`rigid_camera_flow` is an exact depth-conditioned oracle and is not a learned
RGB-only result.

blocker: No trajectory-aligned, calibration-verified depth modality is
available in the public project contract.

next_required_condition: Add or locate a path-free, provenance-bearing depth
adapter with intrinsics, timestamps, pose convention, and visibility masks.
