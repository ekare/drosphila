# Retinotopic motion-field validation

The V1 raw ON/OFF direction-energy tensors were decoded into separate 8x8
direction populations and compared with the exact dense flow on the
real-texture controlled-geometry validation split.

## F0 decision

F0 **failed**. V1-A validation results were:

| Window | Median angular error | Direction sign accuracy | Magnitude Spearman | EPE improvement vs zero |
|---|---:|---:|---:|---:|
| T=3 | 89.14 deg | 0.511 | -0.274 | -2.934 |
| T=5 | 89.74 deg | 0.500 | -0.283 | -2.850 |
| T=7 | 95.20 deg | 0.447 | -0.353 | -2.732 |

The failure is local-field quality, not lack of a global readout. ON and OFF
were decoded separately; their combined result also failed. The field is not
an accepted motion representation and no global SO(3) release candidate was
selected.
