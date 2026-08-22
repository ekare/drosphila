# Evaluation contract v1 report

The v1 evaluator was run with the locked v0.3 checkpoint, trajectory-balanced
sampling, seed 0, legacy-centered intrinsics, and no test-based selection.

| split | samples | mean geodesic | trajectory macro | environment macro | wrong-sign separation median | oracle status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| validation | 512 | 12.2105° | 12.2105° | 12.2105° | -0.00128 | unresolved |
| test | 512 | 13.0248° | 12.9840° | 13.0480° | -0.00106 | unresolved |

The macro intervals are deterministic 1,000-replicate group bootstraps with
seed 0. Validation trajectory/environment 95% intervals are `[12.00, 12.42]`
degrees; test trajectory and environment intervals are `[12.44, 13.46]` and
`[12.62, 13.47]` degrees.

Axis-aware groups were emitted for all six signed dominant axes. The largest
validation mean error was `13.08°` (`-y`); the largest test mean error was
`14.76°` (`+y`). These groups are descriptive and do not change model
selection.

Every sampled terminal state marked temporal agreement unavailable because a
single terminal diagnostic state cannot form a temporal comparison. The
wrong-sign oracle separation failed the contract threshold `0.10` on both
splits. Therefore the real-data directional residual remains diagnostic, not a
correctness oracle.
