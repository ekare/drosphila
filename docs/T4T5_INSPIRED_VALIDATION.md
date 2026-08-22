# T4/T5-inspired cell validation

`scripts/validate_t4t5_cells.py` runs a deterministic engineering check of the
ON and OFF delayed-correlation paths. It includes the requested stationary,
brightness, flicker, edge, bar, grating, patch, aperture, direction, speed,
contrast, nuisance, border, and non-square cases.

The wording is deliberately “T4/T5-inspired”. The measurements are not a
claim of biological equivalence. Border-touching pixels are excluded from the
main statistics by the validity mask; the border-crossing cases remain in the
machine-readable audit.

Example on the 4060 Ti (the local `nvidia-smi` ordinal is operator-specific):

```bash
python scripts/validate_t4t5_cells.py \
  --output-dir /path/to/scratch/t4t5 \
  --scales 4,8,16 \
  --device cuda:1
```

The command writes JSON, Markdown, and a small representative PNG panel. No
dataset or checkpoint is required; the checkpoint field is explicitly
`null` because the audited cell bank is fixed.
