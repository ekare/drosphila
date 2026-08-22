# Scene memory status

Scene memory is not claimed in v0.3 or v0.4. A valid future implementation
needs an explicit keyframe/descriptor contract, causal retrieval, pose or
relative-transform composition, loop-closure evaluation, and a held-out scene
split. It must also distinguish an RGB appearance match from geometric
localization.

The current repository has no learned scene descriptor, map store, loop
closure, or world-relative translation predictor. The depth-conditioned rigid
flow in `TRANSLATION_ORACLE.md` is a prerequisite oracle, not scene memory.
