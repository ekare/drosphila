# GTX 1050 Ti runtime acceptance

Status: **PASS for release-baseline inference/runtime**.

The locked v0.3 checkpoint was loaded and evaluated on CUDA ordinal 0, which
was verified as `NVIDIA GeForce GTX 1050 Ti`. No unrelated desktop process was
stopped.

| workload | result |
| --- | --- |
| batch 1, 128x128, 50 steps | 19.10 ms/sample; 28.57 MiB peak allocated |
| batch 32, 128x128, 50 steps | 2.438 ms/sample; 469.96 MiB peak allocated |
| full validation, 1,697 windows | MSE 0.01934808; geodesic 12.2369° |
| full test, 28,496 windows | MSE 0.02124746; geodesic 12.6353° |

The full 1050 Ti validation/test values match the locked baseline within
normal floating-point variation. This is inference acceptance for the
rotation-only release checkpoint, not acceptance of V1, translation, or scene
memory.

Persistent evidence was written to the operator-provided NV1 scratch disk;
the repository contains only this machine-neutral summary.
