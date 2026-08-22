# Geometric SO(3) solver

The analytic weighted IRLS solver was audited against exact dense homography
flow on the 8x8 field. This isolates the geometry solver from the learned
local field.

| Window | Exact-flow correct error | Exact-flow wrong-sign error | Separation | Median condition |
|---|---:|---:|---:|---:|
| T=3 | 0.0284 deg | 3.8406 deg | 3.8122 deg | 3.008 |
| T=5 | 0.0075 deg | 1.9146 deg | 1.9070 deg | 3.004 |
| T=7 | 0.0038 deg | 1.2759 deg | 1.2721 deg | 3.000 |

The exact-flow oracle is geometrically healthy. Running the same solver on
V1-A decoded local field gives approximately `2.26/1.49/1.38` degrees at
T=3/5/7, but that does not pass F0 because the local field itself has about
90-degree angular error and negative magnitude rank correlation.
