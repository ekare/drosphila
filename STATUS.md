# FlyRot v0.3.0 public status

## Current state

FlyRot is a compact RGB-only research prototype. The v0.2.0 checkpoint uses
three 128x128 RGB frames sampled with a four-frame gap and predicts relative
camera rotation as a three-dimensional rotation vector.

The public checkpoint is `models/flyrot_v0_best.pt`.

## Validation evidence

The accepted checkpoint was evaluated with trajectory-disjoint splits:

- Validation windows: 1,697.
- Validation model MSE: `0.01934808`.
- Validation zero-rotation MSE: `0.01937074`.
- Held-out test windows: 28,496.
- Held-out test model MSE: `0.02124746`.
- Held-out test zero-rotation MSE: `0.02132555`.
- 16 of 17 held-out trajectories improved individually.
- Mean geodesic error: `12.24°` validation and `12.64°` held-out test.

Additional sampling-gap checks also beat the zero-rotation baseline for the
tested short gaps. The full metric table is in
[`reports/evaluation_metrics.md`](reports/evaluation_metrics.md).

The v0.3.0 directional evaluator uses deterministic trajectory-balanced
samples and keeps test data out of model selection:

- Validation: 512 samples, mean geodesic error `12.21°`.
- Test: 512 samples, mean geodesic error `13.02°`.
- Validation/test `directional_residual_pred`: `0.67995` / `0.67990`.
- Validation/test `directional_residual_gt`: `0.68340` / `0.68318`.
- Zero-rotation oracle: `1.0` on both splits.
- Wrong-sign oracle: `0.68207` / `0.68187`.

These are diagnostic baseline measurements, not a claim that the checkpoint
passes a calibrated reliability gate. With three input frames, temporal
agreement is correctly marked unavailable because there are not two real
motion states.

## Known limitations

- The model estimates rotation, not translation or full six-degree-of-freedom
  motion.
- Translation-induced parallax and scene depth are not explicit model inputs.
- `directional_residual_ratio` is an evidence-space direction mismatch, not a
  physical displacement ratio. The legacy `residual_ratio` name is a
  deprecated alias.
- Global reliability and axis confidence are diagnostic scores, not calibrated
  probabilities.
- The training dataset is not included in this repository.

## Reproducibility boundary

Local datasets, generated indices, caches, runs, checkpoints from experiments,
virtual environments, and machine inventories are excluded from the public
repository. All commands in the public documentation use portable paths or
explicit command-line arguments.
