# Diagnostic reliability contract

v0.3.0 separates geometric evidence from confidence language. The diagnostic
path does not turn native direction-cell energy into optical-flow input and it
does not claim a calibrated probability.

## Primary residual

For each pixel and direction, native non-negative evidence is compared with the
direction induced by a candidate SO(3) rotation. The candidate-compatible part
is the positive directional cosine times observed evidence. The primary metric
is

```text
directional_residual_ratio = residual_directional_energy / observed_total_energy
```

It is an evidence-space direction mismatch. It is not a physical displacement
ratio and must not be interpreted as motion magnitude. `residual_ratio` remains
as a deprecated exact alias for v0.2 consumers.

The evaluator reports four controlled values:

- `RR_pred`: model-predicted endpoint rotation;
- `RR_gt`: endpoint ground-truth rotation oracle;
- `RR_zero`: zero-rotation oracle;
- `RR_wrong`: sign-reversed endpoint rotation oracle;
- `delta = RR_pred - RR_gt`.

`RR_gt` is intentionally computed at the final evidence state from the
endpoint target. It is not a fabricated target for every intermediate causal
state.

## Validity semantics

Every optional consistency score has a corresponding validity field.

- Temporal agreement needs two valid, non-warmup motion states. A three-frame
  input has only two direction states and therefore reports temporal agreement
  as unavailable rather than zero.
- Scale agreement compares the separate geometry-evidence vectors and requires
  at least two active scales.
- ON/OFF agreement compares separate ON and OFF geometry-evidence vectors and
  requires both channels to be present.
- Observability uses the information matrix in axis coordinates. Its sorted
  eigenvalues are not mislabeled as `rx`, `ry`, and `rz` confidence.
- Global reliability is a validity-aware geometric aggregation. Invalid
  optional components are excluded, not treated as zero, and the result is
  still a diagnostic score rather than a probability.

Unavailable score fields are represented as `null` in JSON. A zero score means
an available measurement found no agreement; `null` means the measurement was
not defined for that sample.

## Intrinsics and raster transforms

`CameraIntrinsics` stores `(fx, fy, cx, cy, width, height)`. The centered
focal-ratio constructor is retained for legacy data without a calibration
file. A crop subtracts its top-left offset from `(cx, cy)`, and resize scales
both focal lengths and the principal point using pixel-center coordinates.
The full-K path is tested synthetically; a real dataset must provide verified
calibration before its full-K results are claimed.

## Current baseline interpretation

The included v0.2 checkpoint gives balanced v0.3 baseline residuals near 0.68
on both validation and test samples, while the zero oracle is 1.0. The
predicted and endpoint-ground-truth residuals are close, which indicates that
the current native evidence itself is not a reliable correctness oracle. Model
selection therefore requires geodesic/MSE validation gates plus the diagnostic
report, and the test split is reserved for the final report.
