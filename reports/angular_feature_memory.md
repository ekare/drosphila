# Angular feature memory

status: **oracle_only**

evidence: `AngularFeatureMemory` bins features by orientation-aligned rays,
updates mean/second moment/confidence/count/timestamp, and explicitly reports
`renders_pixels: false`. Default 72x18x8 float64 mean/second-moment plus
confidence/count/timestamp storage is 196,992 bytes. Tests pass.

blocker: No accepted orientation stream, re-anchor metric, place retrieval, or
scene split exists.

next_required_condition: Couple the memory to accepted SO(3), evaluate
re-anchoring and retrieval, and keep pixel panorama generation disabled.
