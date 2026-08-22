# Rotation-residual retina report

Status: implemented and tested as a diagnostic branch. The included model
checkpoint is not being presented as a calibrated uncertainty model or as a
translation-robust visual-odometry system.

## What is implemented

`FlyRotV0.forward(..., diagnostics=True)` preserves the default inference
path and additionally returns a rotation-residual bundle. The bundle is
computed after the existing ON/OFF photoreceptor and local direction-cell
stack, so it does not turn native energy into a learned optical-flow target.

The model's direction energy has shape `B,T,C,H,W`, where `C` is eight local
image directions at each of three scales. The observed map is the valid,
non-negative native energy. For each causal state, the predicted rotation
vector is interpreted as a Lie-algebra rotation in radians and converted to
an exact perspective ray displacement:

```text
p' = project(K exp([r]x) K^-1 p)
flow_rotation = p' - p
```

The implementation uses centered normalized coordinates and the configured
`focal_y_over_x` ratio. It is an image displacement in normalized coordinates,
not pixels per second.

The residual remains in energy space. For each direction `d`, with `u_d` the
unit direction-cell vector and `u_R` the predicted rotational-flow direction,
the explained amount is:

```text
E_explained(d,p) = E_observed(d,p) * max(0, dot(u_d, u_R(p)))
E_residual(d,p) = max(0, E_observed(d,p) - E_explained(d,p))
```

The pseudo-flow panels are weighted direction-energy vectors for visualization
only. They must not be described as optical flow or fed back as a new model
input. A first-order rotational-flow basis is used only to build an
observability matrix and its eigenvalue/condition diagnostics; the residual
comparison itself uses the exact perspective projection above.

## Pose convention checked in source

Pose rows are `(tx, ty, tz, qx, qy, qz, qw)`. The target named
`optical_image_motion` is the active ray transform
`R_(t+1)^T R_t` after the source NED-to-optical conversion. Relative rotations
are represented as rotation vectors, not Euler angles. This convention is
also the convention used by the rotational-flow basis and the residual
projection.

## Verification evidence

The clean-code test suite currently reports `29 passed, 2 skipped`, and source
and scripts compile successfully. The geometry tests verify the exact
perspective implementation against the finite-difference first-order basis,
including sign and zero-rotation cases. The model test verifies that
`diagnostics=False` preserves the normal output and that the diagnostic branch
returns finite ON/OFF, residual, axis, and global fields.

The deterministic synthetic smoke test passed all of its gates:

| case | oracle residual ratio | oracle spatial support | interpretation |
| --- | ---: | ---: | --- |
| zero | 1.000 | 0.000 | no motion, no false rotation explanation |
| pure rotation | 0.213 | 1.000 | matching rotation explains most energy |
| brightness | 1.000 | 1.000 | global brightness is not rotation |
| translation | 1.000 | 1.000 | translational field is not rotation |
| moving patch | 1.000 | 0.571 | localized motion remains localized residual |
| mixed | 0.272 | 1.000 | rotation explains the rotational component only |

For the pure-rotation oracle, negating the rotation gives residual ratio
`0.99997`, which is the sign/convention guard. Panels and short RGB videos
are generated below the ignored `artifacts/` directory by
`rotation_residual.py`.

The release checkpoint was also run on six sampled windows from each of the
validation and held-out test splits. These are diagnostic samples, not a full
dataset claim:

| split | samples | mean geodesic | p90 geodesic | residual ratio | spatial support | scale agreement | temporal | ON/OFF | new global |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| validation | 6 | 11.34 deg | 17.42 deg | 0.675 | 0.951 | 0.692 | 0.000 | 0.000 | 0.000 |
| test | 6 | 19.93 deg | 36.35 deg | 0.679 | 0.953 | 0.685 | 0.000 | 0.000 | 0.000 |

The three representative panels for each split are selected as best, median,
and worst by SO(3) geodesic error. The high residual and zero temporal/ON/OFF
agreement are reported rather than hidden: the existing checkpoint's old
magnitude confidence can be close to one while the native motion evidence is
not jointly consistent with the predicted rotation. The new global field is
therefore a conservative, explicitly uncalibrated diagnostic, not a claim of
accuracy.

## Shortcut experiments

On the deterministic pure-rotation sequence, reversing the frame order
reversed the model prediction (cosine to the original `-0.962`) while keeping
its magnitude similar. Repeating the first frame reduced prediction norm from
`0.0458` to `0.0068`, set residual ratio to `1.0`, and reduced old confidence
to `0.334`. Multiplying the frames by `0.5` changed the prediction by only
`0.00033` in rotation-vector norm; multiplying by `2` changed it by `0.00277`.
These are useful invariance/sensitivity checks, not training guarantees.

## Known limitations and next experiments

- The current real checkpoint has a large residual (`about 0.68`) and zero
  temporal/ON/OFF agreement in the sampled reports. That is evidence for a
  model/data or polarity-alignment issue, not evidence that the residual has
  been solved by the old confidence head.
- `global_confidence` and `axis_confidence` are diagnostic scores with no
  calibration protocol. They should not be interpreted as probabilities.
- A translation or independently moving object can produce structured
  residual energy. The diagnostic exposes it; it does not yet learn a robust
  depth/parallax separation model.
- The current observability basis assumes the normalized centered camera grid
  and the configured focal ratio. A camera with a materially different
  principal point or calibration needs an explicit intrinsics extension.
- The next scientific step is to train or adapt the model with the ON/OFF and
  temporal consistency contracts visible in this report, then rerun the full
  trajectory-disjoint validation and held-out test reports. Until that is
  done, this release is a verified diagnostic implementation rather than a
  claim that the checkpoint passes the real-data residual criterion.

## Reproduction

```bash
python scripts/rotation_residual.py --synthetic-smoke-test --save-video
python scripts/rotation_residual_shortcuts.py
python scripts/rotation_residual.py \
  --checkpoint models/flyrot_v0_best.pt \
  --index artifacts/tartanair2_index.json \
  --split validation --sample-count 6 --preload-images
python scripts/rotation_residual.py \
  --checkpoint models/flyrot_v0_best.pt \
  --index artifacts/tartanair2_index.json \
  --split test --sample-count 6
```

All dataset indices, generated panels, videos, reports, caches, and local
configuration remain outside the public source tree.
