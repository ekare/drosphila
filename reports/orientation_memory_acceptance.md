# Orientation memory acceptance

status: **oracle_only**

evidence: `SO3OrientationBelief` composes ordered exponential-map updates,
rejects invalid/low-confidence measurements, retains covariance, and passes
the non-commuting composition/rejection tests. It is full SO(3) state, not a
1-D ring. No predicted phone rotation stream passes the upstream field gate.

blocker: F0-DIRECTION and learned rotation acceptance are failed.

next_required_condition: Feed only an accepted relative-rotation estimator,
then measure drift, reset/re-anchor, and confidence behavior on held-out
trajectories.
