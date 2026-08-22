# Evaluation contract v1

The evaluator is a measurement contract, not a model-selection shortcut.
Selection uses validation only; the held-out test is reported once the
configuration is locked.

## Required measurements

- RGB-only model metrics versus a zero-rotation baseline.
- Trajectory-balanced sampling and trajectory-disjoint splits.
- Mean/median/p90 geodesic error, component and magnitude error.
- Axis-aware bins (`+x`, `-x`, `+y`, `-y`, `+z`, `-z`, `near_zero`).
- Environment and trajectory macro means with deterministic group bootstrap
  intervals.
- Directional residuals for prediction, ground truth, zero, and wrong sign.
- Explicit validity fields and reason codes. Unavailable temporal or
  observability quantities are not converted to numeric zero evidence.

## Oracle gate

The synthetic exact-homography gate must separate the correct sign from the
wrong sign. The current real-data diagnostic uses a required median
wrong-sign-minus-ground-truth residual separation of `0.10`. If that is not
met, the report status is `unresolved` and the residual is diagnostic only.

## Translation boundary

Rotation-only residuals cannot establish translation correctness in scenes with
depth variation. A translation claim requires a depth-verified scope, full pose
projection, parallax grouping, and an independent held-out evaluation. The
current fast-cache RGB+pose index has no depth records, so its baseline cannot
pass that gate.
