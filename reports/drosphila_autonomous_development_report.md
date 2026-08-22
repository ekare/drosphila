# Drosphila Autonomous Development Report

## 1. Executive Verdict

The v0.3.0 RGB-only rotation baseline is reproducible and accepted for
GTX-1050-Ti inference. Pose/image geometry and provenance contracts now pass.
FlyRot-v1 is implemented experimentally but is not accepted: held-out test
metrics are slightly worse and the real directional oracle remains unresolved.
Translation is oracle geometry only; scene memory is not implemented.

## 2. Repository and Starting Point

Work started from the public `v0.3.0` tag at `ac177dce` on a separate branch
`codex/v0.4.0-visual-motion-foundation`. The existing v0.3 release branch and
checkpoint were preserved.

## 3. Provenance and Privacy

Portable run-manifest helpers record commit, seed, command, software, logical
dataset manifest hash, and artifact hashes. Absolute paths, usernames,
hostnames, IPs, GPU UUIDs, and local configs are excluded from tracked reports.
The staged-content scan found no local path/host/IP/UUID matches.

## 4. Dataset and Split Contract

The locked logical dataset is TartanAir V2-style `lcam_front`, RGB+pose,
84 trajectory-camera records. The trajectory-disjoint split has 13 train
records/8,536 windows, 2 validation records/1,697 windows, and 17 test
records/28,496 windows. The fast-cache index has zero indexed depth records.

## 5. Pose and Image-Motion Convention

Pose rows are `tx ty tz qx qy qz qw` in TartanAir NED. The relative rotation is
`R_t.T @ R_t+1`; image motion takes its transpose; optical conversion uses the
explicit NED-to-right/down/forward basis. These rules are executable and tested.

## 6. Camera Intrinsics and Raster Geometry

Full `fx, fy, cx, cy` intrinsics, non-square/off-center projection, and
crop/resize propagation are implemented. The public TartanAir V2 `f=320,
(cx,cy)=(320,320)` calibration is tracked as `assumed` pending exact-manifest
verification. The legacy 128 raster and resized assumption differ by at most
0.068 pixels for the recorded sensitivity case.

## 7. v0.3 Baseline Identity

The locked model is `flyrot_motion_direct`, 39 parameters, RGB 128x128,
three frames, frame gap four, target `optical_image_motion`. Config, checkpoint,
and path-free dataset manifest hashes are in `baseline_v0.3.0_locked.json`.

## 8. v0.3 Baseline Metrics

Validation: MSE `0.01934808` versus zero `0.01937074`, mean geodesic `12.2369°`.
Held-out test: MSE `0.02124746` versus zero `0.02132555`, mean geodesic
`12.6353°`. The test was reported after selection and not used for selection.

## 9. Directional Residual Oracle

Synthetic exact-rotation cases pass the sign guard. Real balanced validation
has `RR_pred=0.67995`, `RR_gt=0.68340`, `RR_wrong=0.68207`; test has
`0.67990`, `0.68318`, `0.68187`. Correct and wrong signs do not separate, so
the residual is diagnostic and not a learned correctness oracle.

## 10. Evaluation Contract

The evaluator now emits axis bins, trajectory/environment macro means with
deterministic group bootstrap intervals, and validity reason codes. Unavailable
temporal agreement is represented as unavailable, never as zero evidence.

## 11. Synthetic Geometry Gate

Exact homography golden tests pass at `1e-10` tolerance for non-square,
off-center intrinsics. Correct-sign displacement has positive median dot
product and wrong-sign displacement negative median dot product. Rigid depth
flow decomposition and scale-free translation direction also pass synthetic
tests.

## 12. FlyRot-v1 Architecture

V1 is a separate model with explicit ON/OFF evidence paths, an 8x8 retinotopic
field, and per-step outputs. It uses matrix SO(3) composition rather than
component-wise angle accumulation. It has 829 trainable parameters.

## 13. FlyRot-v1 Synthetic Result

In the pure-rotation homography training smoke, step 200 reached MSE `0.0191944`
versus zero `0.0192742`. This is an architecture smoke result, not a release
or generalization claim.

## 14. FlyRot-v1 Real-Data Result

Validation-best step 200 reached full-validation MSE `0.0193121` versus zero
`0.0193707`. On held-out test it reached MSE `0.0212693` and geodesic `12.6455°`,
slightly worse than v0.3. V1 is therefore not promoted or released.

## 15. Orientation Memory

Ordered per-step SO(3) matrix composition is implemented and passes a
non-commuting two-step test. It remains experimental because V1 did not pass
the held-out generalization gate.

## 16. Translation and Scale

Depth-conditioned rigid camera flow, rotational/parallax decomposition, and a
scale-free translation direction helper are implemented as oracle geometry.
No RGB-only translation predictor or metric-scale claim is made.

## 17. Scene Memory

No scene descriptor, keyframe map, causal retrieval, loop closure, or
world-relative learned translation exists. Scene memory is explicitly not
started rather than inferred from rotation-only results.

## 18. GTX 1050 Ti Runtime

CUDA ordinal 0 was verified as GTX 1050 Ti. Batch-1 128x128 inference measured
`19.10 ms/sample` and `28.57 MiB`; batch-32 measured `2.438 ms/sample` and
`469.96 MiB`. Full 1050 validation/test matched the locked baseline.

## 19. Storage and Process Safety

The dataset remained read-only. Hot evidence/checkpoints were persisted under
the operator-provided NV1 scratch disk; no unrelated desktop process was
stopped. The root filesystem was monitored because it was near capacity.

## 20. Tests and Build

Local verification: `47 passed, 2 skipped`; Python compilation passed; wheel
build and wheel-content check passed; CI synthetic smoke passed. GitHub Actions
also passed on Python 3.11 and 3.12.

## 21. Git and Release State

The v0.3.0 tag/release remains unchanged. The v0.4 branch has logical commits
for provenance, geometry, evaluation, V1, translation oracle, memory gates,
and runtime acceptance. No v0.4 release tag was created because the learned
quality gates are not all satisfied.

## 22. Pull Request and Next Blocker

PR #2 is open at `https://github.com/ekare/drosphila/pull/2`. The next blocker
is a depth-verified, trajectory-disjoint translation/scene dataset contract;
the secondary blocker is a real visual-motion oracle that separates correct
and wrong signs.

## 23. Machine-Readable Summary

The canonical machine-readable summary is
[`drosphila_autonomous_development_summary.json`](drosphila_autonomous_development_summary.json).
Its required fields distinguish verified, experimental, missing, and
unresolved capabilities.
