# FlyRot-v0 evaluation summary

This report describes the accepted RGB-only checkpoint. Dataset files and
machine-specific paths are intentionally not part of the report.

## Main split

| split | windows | model MSE | zero-rotation MSE | mean geodesic |
| --- | ---: | ---: | ---: | ---: |
| trajectory-disjoint validation | 1,697 | 0.01934808 | 0.01937074 | 12.2369° |
| held-out test | 28,496 | 0.02124746 | 0.02132555 | 12.6353° |

The held-out test contains 17 trajectories. The model improved over the
zero-rotation baseline on 16 of 17 trajectories.

Validation geodesic median and p90 are `11.1653°` and `20.8984°`.
Held-out test geodesic median and p90 are `11.1973°` and `22.3765°`.

## Additional metrics

- Validation component MAE: `[6.4888°, 6.1892°, 5.8777°]`.
- Held-out test component MAE: `[6.3663°, 6.6969°, 6.0925°]`.
- Validation direction accuracy: `51.80%`.
- Held-out test direction accuracy: `53.64%`.
- Validation axis cosine: `0.0370`.
- Held-out test axis cosine: `0.0493`.

These directional metrics are weak; the main acceptance claim is the small but
repeatable aggregate improvement over the zero-rotation baseline.

## Confidence proxy

The checkpoint exposes a magnitude-based RGB-only confidence proxy. It does
not change rotation weights and should not be interpreted as calibrated
uncertainty.

At 25% selective coverage, the mean geodesic risk was `11.6166°` on
validation and `11.8216°` on held-out test, compared with full-set risks of
`12.2369°` and `12.6353°`.

## Scope

Inputs are RGB frames only. Pose is used for training and evaluation targets;
depth, IMU, externally computed optical flow, and semantic labels are not
inference inputs. Translation-induced parallax remains a known limitation.
