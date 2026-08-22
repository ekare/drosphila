# Exact real-texture pure-rotation benchmark

`scripts/run_real_texture_benchmark.py` reads RGB frames from an operator
supplied fast-cache index and generates an online, deterministic controlled
geometry benchmark. It does not copy or modify the source dataset.

Each example records a path-free logical source image ID, resized intrinsics,
exact per-step and endpoint SO(3), dense homography flow, valid warp mask,
crop/resize metadata, texture observability statistics, and seed. Train,
validation, and test use trajectory-disjoint records. Test metrics are
reported only after train/validation generation and are never used for model
selection.

The benchmark is called **real-texture controlled geometry**, not synthetic
texture. It uses

```text
H = K_target @ R_image @ inverse(K_source)
```

and explicitly reports the correct-sign versus wrong-sign geometric oracle.

Example:

```bash
python scripts/run_real_texture_benchmark.py \
  --index artifacts/tartanair2_kiousb_lcam_index.json \
  --output-dir /path/to/scratch/real-texture \
  --device cuda:1
```
