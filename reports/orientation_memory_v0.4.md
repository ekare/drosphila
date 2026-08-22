# Orientation memory status

Status: **implemented experimentally; not release-validated**.

The v1 candidate composes per-step SO(3) matrices in order and exposes both
step and cumulative outputs. The non-commuting two-step synthetic test passed.
The candidate had 829 parameters and stayed below the 50k budget.

Its full held-out test MSE was `0.0212693` versus the locked v0.3 baseline
`0.0212475`; mean geodesic was `12.6455°` versus `12.6353°`. The orientation
memory implementation is therefore retained as an experimental branch of the
architecture, not promoted to a release capability.
