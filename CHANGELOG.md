# Changelog

## 0.3.0

- Added a validity-aware rotational-evidence diagnostic and exact
  `directional_residual_ratio` semantics.
- Kept `residual_ratio` as a deprecated compatibility alias.
- Added full camera intrinsics with crop/resize transforms and geometry tests.
- Added trajectory-balanced oracle evaluation with `RR_pred`, `RR_gt`,
  `RR_zero`, `RR_wrong`, signed `delta`, distributions, groups, and
  risk/coverage summaries.
- Added expanded deterministic shortcut checks and a negative scale-response
  calibration audit.
- Fixed packaging of `flyrot.data` and added Python 3.11/3.12 CI, wheel
  validation, and dataset-free synthetic smoke testing.
- Added MIT licensing and machine-neutral reproducibility documentation.
