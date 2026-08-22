# Rotation-residual worklog

This is a public, machine-neutral worklog. It intentionally omits usernames,
hostnames, absolute paths, GPU identifiers, dataset roots, and local index
contents.

## 2026-08-22

- Inspected the existing photoreceptor, direction-cell, rotational-evidence,
  pose-convention, and first-order rotational-flow implementations.
- Confirmed that the existing model output is causal accumulated rotation
  evidence with shape `B,T,3`; it is not optical flow.
- Added exact perspective SO(3) ray projection and a native energy-space
  residual decomposition under `flyrot.diagnostics`.
- Added ON/OFF component exposure behind the opt-in
  `FlyRotV0(..., diagnostics=True)` flag. The default forward API remains
  unchanged.
- Added residual maps, exact predicted rotation flow, diagnostic pseudo-flow,
  scale residuals, spatial support, scale/temporal/ON/OFF agreement,
  observability eigenvalues, and uncalibrated confidence fields.
- Added deterministic synthetic cases for zero motion, rotation, brightness,
  translation, moving patch, and mixed motion. The synthetic acceptance gates
  passed, including the wrong-rotation sign guard.
- Added headless PNG panels and short RGB videos. Added shortcut experiments
  for reversed frames, repeated frames, and brightness scaling.
- Added real-data validation and held-out test sampling with best/median/worst
  diagnostic panels. The sampled checkpoint results are recorded in the
  public report as evidence, not as a full-dataset claim.
- Added unit coverage for exact perspective flow, native residual behavior,
  zero-energy confidence, and the opt-in model diagnostics API.
- Verification: `29 passed, 2 skipped`; Python compilation completed without
  errors.

## Interpretation

The oracle geometry behaves as required: matching pure rotation leaves a
small residual, the wrong sign leaves nearly all energy unexplained, and
brightness/translation/moving-object energy is not labeled as camera
rotation. The current release checkpoint still has a large real-data
residual and zero temporal/ON/OFF agreement in the sampled windows. The next
model-development step is therefore consistency-aware training/calibration,
not a claim that the old confidence output is sufficient.
