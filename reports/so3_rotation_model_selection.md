# SO(3) rotation model selection

status: **oracle_only**

evidence: Exact SO(3) exponential/logarithm and robust weighted solver
primitives are covered by geometry tests and the prior geometric solver
report. No learned phone field currently supplies an accepted input to this
solver.

blocker: F0-DIRECTION is failed.

next_required_condition: Obtain a corrected local field that passes direction,
sign, and observability gates, then compare the analytic solver with a small
learned readout without using test data for selection.
