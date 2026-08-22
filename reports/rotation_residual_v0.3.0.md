# v0.3.0 rotation-residual and oracle report

The report is for the included v0.2.0 checkpoint and the v0.3.0 diagnostic
implementation. It uses trajectory-balanced samples and keeps the test split
out of model selection.

## Contract checks

The deterministic synthetic suite passed all cases: zero motion, pure
rotation, brightness-only change, translation, localized moving patch, and
mixed motion. Pure-rotation oracle residual was `0.213`; the sign-reversed
oracle residual was `0.99997`. Non-rotational cases remained unexplained by
the oracle. Full intrinsics, crop/resize transforms, validity masks, and JSON
null serialization are covered by unit tests.

## Real-data baseline

The evaluator used three 128x128 frames, a four-frame gap, the
`optical_image_motion` target convention, and 512 deterministic samples
balanced across trajectories.

| split | mean geodesic | p90 geodesic | RR_pred | RR_gt | RR_zero | RR_wrong | delta mean | temporal valid |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| validation | 12.21° | 20.73° | 0.67995 | 0.68340 | 1.00000 | 0.68207 | -0.00345 | 0% |
| test | 13.02° | 23.19° | 0.67990 | 0.68318 | 1.00000 | 0.68187 | -0.00329 | 0% |

Scale, ON/OFF, observability, and global-reliability fields were valid for all
these samples. Temporal validity is zero by design for a three-frame window:
there are not two post-warmup motion comparisons. This is an unavailable
measurement, not a zero score.

The predicted and endpoint-ground-truth residuals are nearly equal. Combined
with the zero oracle at 1.0, this says the current native evidence does not
provide a trustworthy correctness oracle. The old magnitude confidence is
therefore not promoted to calibrated uncertainty.

## Shortcut results

The expanded deterministic shortcut suite covered time reversal and shuffling,
frame repetition, brightness and gamma changes, texture loss, center/edge
localization, a small independent patch, color change, principal-point/focal
perturbations, sign reversal, axis permutation, and an oracle reference.

- Reversing time changed the prediction cosine to `-0.962` while preserving
  nearly the same norm.
- Repeating one frame reduced prediction norm from `0.04576` to `0.00676`,
  made the residual `1.0`, and made global reliability invalid.
- Brightness x0.5 changed the prediction by only `0.000329` in rotation-vector
  norm; brightness x2 changed it by `0.00277`.
- The pure-rotation oracle residual was `0.213`, while the learned model on
  the same synthetic sequence was about `0.667`.

These are diagnostic invariance and failure-mode checks, not generalization
claims.

## Reproduction

```bash
python scripts/rotation_residual.py --synthetic-smoke-test --output artifacts/synthetic
python scripts/rotation_residual_shortcuts.py --output artifacts/shortcuts
python scripts/evaluate_directional_residual.py \
  --checkpoint models/flyrot_v0_best.pt \
  --index artifacts/tartanair_index.json \
  --split validation --samples 512 \
  --output artifacts/directional_validation.json
```

The index and generated artifacts are local and ignored by Git.
