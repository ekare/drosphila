# Drosphila T4/T5 Motion-Field Validation Report

## 1. Executive Verdict

Stable v0.3.1 is released with no new learned weights. FlyRot-v1 remains
implemented, experimental, and not accepted. C0 and F0 failed; v0.4.0 was not
released.

## 2. PR and Release State

PR #3 merged as `970400abafa1273ea1674f73223a5a6e3c055d5b`; tag/release v0.3.1
exists. PR #2 is open only because its mixed stable/experimental scope is a
release gate. PR #4 tracks the experimental line.

## 3. T4/T5 Cell-Level Validation

The deterministic requested grid ran on the 4060 Ti. Current scales failed C0;
the one bounded small-motion correction also failed.

## 4. Polarity and Direction Tuning

ON/T4-inspired and OFF/T5-inspired paths were reported separately. Direction
sign and polarity cross-talk fail the engineering thresholds.

## 5. Speed and Scale Audit

The current `(4,8,16)` bank under-covers low displacement. `(1,2,4,8)` improves
selectivity but not direction sign. No further search was allowed.

## 6. Real-Texture Pure-Rotation Dataset

Online exact homography generation preserves real RGB texture, uses disjoint
train/validation/test trajectories, and records exact K, SO(3), flow, masks,
observability, and seed.

## 7. Local Motion-Field Accuracy

F0 failed: validation angular error is about 89–95 degrees, sign accuracy
0.45–0.51, and magnitude Spearman is negative for T=3/5/7.

## 8. Geometric SO(3) Solver

The analytic solver separates correct and wrong sign on exact dense flow and is
not the primary blocker. It receives a failed learned local field in V1-A.

## 9. Depth/Flow Dataset Inventory

The active 84-record fast-cache has no depth, flow, or segmentation. A separate
teacher-depth precompute exists but is not metric/type/calibration verified.

## 10. Depth-Backed Real Oracle

Unresolved. No depth-backed acceptance claim is made.

## 11. Controlled Model Candidates

V1-A and V1-B were measured diagnostically; V1-C was not started because F0
failed. No candidate winner exists.

## 12. Three-Seed Validation

Not run: a single candidate was not eligible after C0/F0 failure.

## 13. Held-Out Test

The real-texture test split was evaluated only as a post-selection diagnostic;
it was not used for selection. There is no locked winner test claim.

## 14. GTX 1050 Ti Runtime

V1-A is 23.54/26.01/27.68 ms per sample at T=3/5/7 and remains below the
runtime budget. Runtime success does not override quality failure.

## 15. Acceptance Gates

C0: failed. F0: failed. G1-CONTROLLED: not run as an acceptance gate.
v0.4.0: not released.

## 16. Failed Experiments

The failures are labeled `cell_direction_failure`, `polarity_failure`, and
`local_field_failure`; pose/intrinsics and teacher-depth verification remain
additional unresolved boundaries.

## 17. Commits, PRs, Tags and Releases

Stable merge: `970400a`. Experimental validation: `930c0c1`. Release: v0.3.1.
No v0.4.0 tag or release.

## 18. Remaining Scientific Blocker

The retinotopic local direction/polarity field is not correct on deterministic
stimuli or exact real-texture rotation. Global SO(3) tuning must not hide this.

## 19. Exact Reproduction Commands

```bash
python scripts/validate_t4t5_cells.py --output-dir SCRATCH/t4t5 --scales 4,8,16 --device cuda:1
python scripts/validate_t4t5_cells.py --output-dir SCRATCH/t4t5-small --scales 1,2,4,8 --device cuda:1
python scripts/run_real_texture_benchmark.py --index artifacts/tartanair2_kiousb_lcam_index.json --output-dir SCRATCH/real-texture --device cuda:1
python scripts/audit_geometric_solver.py --index artifacts/tartanair2_kiousb_lcam_index.json --output-dir SCRATCH/solver --device cuda:1
pytest -q
```

## 20. Machine-Readable Summary

See `drosphila_t4t5_motion_field_validation.json` for the locked status,
metrics, hashes, release state, runtime, and next blocker.
