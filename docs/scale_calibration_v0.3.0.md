# Direction-scale calibration v0.3.0

The deterministic calibration sweep covers `rx/ry/rz`, positive and negative
angles, four angle magnitudes, four spatial regions, two texture seeds, two
contrast factors, three intrinsic configurations, and the configured frame
gap. It produced 3,456 scale rows.

The experiment intentionally tests whether native scale response can be used
as a physical displacement magnitude. The current synthetic direction oracle
is direction-quantized and repeated over the three scales, so it is a useful
negative control rather than a learned magnitude target.

| scale | evidence norm range | mean direction accuracy | Spearman evidence vs normalized displacement | Spearman evidence vs pixel displacement |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.403–3.044 | 0.701 | -0.231 | -0.217 |
| 2 | 0.403–3.044 | 0.701 | -0.231 | -0.217 |
| 3 | 0.403–3.044 | 0.701 | -0.231 | -0.217 |

The scales therefore do not provide a reliable physical displacement
calibration in this release. `v0.3.0` keeps the metric directional and does
not enable a `magnitude_aware_residual_ratio`. The complete machine-neutral
JSON summary is generated at `reports/scale_calibration_v0.3.0.json`; the
larger row artifact remains under ignored `artifacts/scale_calibration/`.

Reproduce with:

```bash
python scripts/calibrate_direction_scales.py \
  --output reports/scale_calibration_v0.3.0.json \
  --artifact-dir artifacts/scale_calibration
```
