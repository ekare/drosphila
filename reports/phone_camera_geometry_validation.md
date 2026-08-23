# Phone camera geometry validation

status: **accepted** for the input/geometry contract only.

evidence: `tests/test_phone_camera_contract.py` passed 6 tests. The contract
covers arbitrary aspect ratios, off-center principal points, crop/resize
history, Brown-Conrady round-trip, timestamp/exposure/rolling-shutter
metadata validation, finite ray grids, and angular/pixel displacement
conversion. Production rasters are 320x180, 256x144, and 192x108; 128x128 is
compatibility-only.

The contract does not accept the learned motion field or any biological
equivalence claim.
