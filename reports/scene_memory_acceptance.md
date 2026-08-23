# Scene memory acceptance

status: **not_started**

evidence: No learned descriptor, keyframe store, causal retrieval, loop
closure verifier, or world-relative translation predictor is present. The
angular feature memory is not a scene map and does not render pixels.

blocker: Upstream orientation/translation are not accepted.

next_required_condition: Implement a compact keyframe/descriptor/relative-edge
prototype only after the pose inputs pass, then report precision, recall,
false closures, and drift before/after.
