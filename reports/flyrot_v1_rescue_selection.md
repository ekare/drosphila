# FlyRot-v1 rescue selection

## Candidate set

- **V1-A:** existing 829-parameter FlyRot-v1 with learned head; checkpoint
  SHA256 `3d26a94a50a30a5b95bc30f9e982e045b201c88b8d5d3e8e4e2409bb49bcf81f`.
- **V1-B:** decoded polarity-separated field plus analytic weighted SO(3)
  solver. Exact-flow geometry passes, learned-field F0 fails.
- **V1-C:** V1-B plus a learned residual. Not started because F0 failed and
  adding a residual would hide the local-field failure.

## Selection decision

No winner was selected. The one-seed V1-A diagnostic was not eligible for
selection: validation local angular error is about 89–95 degrees, direction
sign accuracy is 0.45–0.51, and magnitude Spearman is negative. No test split
was used to choose a candidate. No unbounded hyperparameter search or global
rotation training was started after the C0/F0 failures.
