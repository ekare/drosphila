# Retinotopic motion-field acceptance

Evidence class: PROJECT_EVIDENCE. The unchanged V1-A checkpoint was evaluated
on real TartanAir RGB texture warped by exact pure rotations under the
normal-phone raster contract. No training or test-based selection was used.

## Result

The F0 direction gate failed on all phone rasters and contexts. Representative
validation results are:

| Raster | Context | Median angular error | Direction sign | EPE improvement |
|---|---:|---:|---:|---:|
| 192x108 | T3 | 67.22 deg | 0.631 | -4.93 |
| 192x108 | T5 | 55.19 deg | 0.609 | -2.56 |
| 256x144 | T3 | 89.09 deg | 0.488 | -5.61 |
| 256x144 | T5 | 82.44 deg | 0.511 | -4.13 |
| 320x180 | T3 | 86.28 deg | 0.511 | -4.89 |
| 320x180 | T5 | 86.63 deg | 0.515 | -3.76 |

T9 has very low observable-cell coverage and remains poor. T17 has no
observable cells in this bounded run and is not an accepted temporal
solution. Magnitude is not accepted; no physical-pixel claim is made.

## Gate decision

F0-DIRECTION: failed
F0-MAGNITUDE: unavailable_un_calibrated
F0-EPE: failed
F0-OBSERVABILITY: failed_or_insufficient_for_T9_T17

This is a local motion-field failure, not evidence that a normal phone FOV is
fundamentally impossible. A separate 60/90/120 degree observability diagnostic
is still required before assigning the failure to FOV.

Raw machine-readable evidence is stored in the operator scratch artifact named
phone_motion_field_audit.json.
