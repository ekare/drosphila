# FlyRot-v1 candidate report

Status: **experimental; not accepted as a release model**.

## Architecture gate

- Trainable parameters: `829` (`<50,000` budget)
- Separate ON and OFF direction-evidence paths: implemented and tested
- Retinotopic field: separate ON/OFF `8x8` field per step, implemented and
  returned by the model
- Orientation memory: ordered per-step SO(3) matrices and cumulative matrix
  composition, implemented and tested
- Synthetic pure-rotation training smoke: validation MSE `0.0191944` versus
  zero `0.0192742` at step 200

The synthetic result is a smoke result, not a generalization or release gate.

## Real-data candidate selection

Selection was made on a short trajectory-disjoint validation subset only. The
best candidate was step 200; step 300 was not selected because its subset MSE
returned above zero.

| model | validation scope | model MSE | zero MSE | test MSE | test geodesic |
| --- | --- | ---: | ---: | ---: | ---: |
| v0.3 locked baseline | full | 0.01934808 | 0.01937074 | 0.02124746 | 12.6353° |
| v1 candidate, step 200 | 512 subset / full report | 0.01920698 / 0.01931209 | 0.01927422 / 0.01937074 | 0.02126930 | 12.6455° |

The v1 full validation MSE is lower than v0.3, but held-out test MSE and
geodesic error are slightly worse. The candidate therefore fails the
generalization/release gate. Its confidence is constant (`std=0`), so it has
no usable selective uncertainty behavior.

## Geometry diagnostic

On a 512-sample trajectory-balanced validation report, v1 had
`RR_pred=0.68073`, `RR_gt=0.68327`, and `RR_wrong=0.68202`; wrong-sign median
separation was `-0.00043` against the required `0.10`. The oracle status is
`unresolved`, consistent with the baseline. This is why the candidate is not
released despite the small validation improvement.

The experimental class remains useful as an isolated architecture for future
oracle-gated work. No candidate checkpoint is committed to the public model
directory.
