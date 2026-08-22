# FlyRot v0.3.0 locked baseline

This is the reproducibility lock for the released RGB-only rotation baseline.
It records the rerun from a clean `v0.3.0` checkout. Dataset files and local
filesystem paths are intentionally excluded.

## Provenance

- Measurement commit: `ac177dceae2c53b3de88bf13cc66f858828a7f07`
- Measurement worktree: clean
- Seed: `0`
- Dataset: TartanAir V2-style `lcam_front`, logical fast-cache index
- Path-free dataset records: `84`
- Path-free dataset manifest SHA-256: `e95a55ee71ed7b5d6df3efc5bb16613904456f3467c49667365feccd53c25853`
- Config SHA-256: `873b041b51fe1c855d8eb4337cdc0bde64bda287d744ca9751386b8bb7afbebf`
- Checkpoint SHA-256: `f8832b0215f02bec529ad0236bc8fb1967a09ca2102fa3f54b1a3f719c4b7c48`
- Runtime: Python 3.11, PyTorch 2.5.1+cu121, CUDA runtime 12.1
- Model: `flyrot_motion_direct`, 39 trainable parameters
- Input: RGB, 128x128, 3 frames, frame gap 4
- Target: `optical_image_motion`
- Intrinsics mode: centered legacy mode; the TartanAir V2 camera calibration
  is still an explicit assumption in this baseline.

## Commands

```text
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider
.venv/bin/python -m compileall -q src scripts
.venv/bin/python scripts/evaluate_checkpoint.py --checkpoint models/flyrot_v0_best.pt --index artifacts/tartanair2_kiousb_lcam_index.json --split validation --samples-per-record 4096 --image-size 128 --window-length 3 --frame-gap 4 --target-direction optical_image_motion --batch-size 32 --preload-images --preload-workers 8
.venv/bin/python scripts/evaluate_checkpoint.py --checkpoint models/flyrot_v0_best.pt --index artifacts/tartanair2_kiousb_lcam_index.json --split test --samples-per-record 4096 --image-size 128 --window-length 3 --frame-gap 4 --target-direction optical_image_motion --batch-size 32 --preload-images --preload-workers 8
```

The full held-out test was evaluated without using it for model selection.

## Results

| split | windows | model MSE | zero MSE | mean geodesic | median | p90 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| validation | 1,697 | 0.01934808 | 0.01937074 | 12.2369° | 11.1653° | 20.8984° |
| held-out test | 28,496 | 0.02124746 | 0.02132555 | 12.6353° | 11.1973° | 22.3765° |

Balanced directional diagnostics remain unresolved: validation
`RR_pred=0.67995`, `RR_gt=0.68340`, `RR_wrong=0.68207`; test
`RR_pred=0.67990`, `RR_gt=0.68318`, `RR_wrong=0.68187`. These values do not
separate correct and incorrect rotation strongly enough to serve as a learned
correctness oracle. Temporal validity is unavailable for the single terminal
diagnostic state and must not be interpreted as a score of zero.

## Scope verdict

The checkpoint is a reproducible rotation-only release baseline. It is not a
validated visual-motion oracle, does not estimate translation, and does not
support a claim of calibrated confidence or world-relative motion.
