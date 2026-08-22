# v0.3.0 controlled model selection

Six candidates were trained with the same trajectory-disjoint split, seed,
input raster, frame gap, target convention, loss, 300-step budget, and
validation budget. Each used a locally supplied fast-cache index. The test
split was not used for selection. No candidate was accepted because none beat
the zero-rotation validation MSE in this short controlled budget.

| candidate | validation MSE | zero MSE | validation geodesic | validation RR_pred | test geodesic | test RR_pred | peak VRAM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| motion_direct | 0.01937533 | 0.01927422 | 12.2585° | 0.679090 | 13.0640° | 0.679220 | 1.85 GB |
| scale_separated_gated | 0.02063351 | 0.01927422 | 12.6962° | 0.679848 | 13.6469° | 0.680109 | 1.85 GB |
| lstsq_gated | 0.04019848 | 0.01927422 | 17.1269° | 0.678628 | 17.7412° | 0.678587 | 1.85 GB |
| appearance_gated | 0.01932106 | 0.01927422 | 12.2536° | 0.674778 | 13.1906° | 0.671382 | 1.35 GB |
| motion_consistency | 0.01934027 | 0.01927422 | 12.2439° | 0.679324 | 13.0508° | 0.679470 | 1.85 GB |
| motion_uncertainty | 0.01940124 | 0.01927422 | 12.2604° | 0.677510 | 13.0739° | 0.677407 | 1.85 GB |

The existing checkpoint remains the release baseline: its fast-cache-backed
balanced validation geodesic was `12.2105°`, better than every short candidate.
Candidate checkpoints remain local generated artifacts and are not included in
the release. Their full oracle distributions are available by rerunning
`scripts/evaluate_experiment_matrix.py`.

## GPU load note

The controlled matrix used batch-32 for comparability. A separate batch-256
throughput probe reached the device memory limit and failed with CUDA OOM. A
batch-128 probe completed with peak VRAM `7.37 GB` and no OOM. These probes do
not affect model selection.
