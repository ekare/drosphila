# Runtime model comparison

GTX 1050 Ti, fixed 128x128 input, batch 1, 120 warm repeated forwards:

| Model | Params | T=3 ms/sample | T=5 ms/sample | T=7 ms/sample | T=3 peak MiB | T=5 peak MiB | T=7 peak MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| v0.3 baseline | 39 | 18.96 | 19.75 | 18.36 | 28.6 | 38.0 | 49.2 |
| V1-A | 829 | 23.54 | 26.01 | 27.68 | 34.6 | 59.6 | 85.3 |

CPU smoke, finite outputs, serialization round-trip, and fixed-input CPU/CUDA
parity passed for both checkpoints. Maximum absolute parity error was below
`1.2e-5`. V1-A remains experimental despite meeting the runtime budget.
