# FlyRot-v0 public status

## Current state

FlyRot-v0 is a compact RGB-only research prototype. The accepted model uses
three 128x128 RGB frames sampled with a four-frame gap and predicts relative
camera rotation as a three-dimensional rotation vector.

The public checkpoint is `models/flyrot_v0_best.pt`.

## Validation evidence

The accepted checkpoint was evaluated with trajectory-disjoint splits:

- Validation windows: 1,697.
- Validation model MSE: `0.01934808`.
- Validation zero-rotation MSE: `0.01937074`.
- Held-out test windows: 28,496.
- Held-out test model MSE: `0.02124746`.
- Held-out test zero-rotation MSE: `0.02132555`.
- 16 of 17 held-out trajectories improved individually.
- Mean geodesic error: `12.24°` validation and `12.64°` held-out test.

Additional sampling-gap checks also beat the zero-rotation baseline for the
tested short gaps. The full metric table is in
[`reports/evaluation_metrics.md`](reports/evaluation_metrics.md).

## Known limitations

- The model estimates rotation, not translation or full six-degree-of-freedom
  motion.
- Translation-induced parallax and scene depth are not explicit model inputs.
- The current confidence value is a practical magnitude-based abstention
  proxy, not a calibrated probability.
- The training dataset is not included in this repository.

## Reproducibility boundary

Local datasets, generated indices, caches, runs, checkpoints from experiments,
virtual environments, and machine inventories are excluded from the public
repository. All commands in the public documentation use portable paths or
explicit command-line arguments.
