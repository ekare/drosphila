# Motion scale audit

The current cell bank uses scales `(4, 8, 16)` pixels per step. The
deterministic stimulus grid shows weak/absent matched response at the
0.5/1/2-pixel regime and does not provide reliable direction sign across the
requested directions. The bounded small-motion candidate `(1,2,4,8)` improves
selectivity but not direction sign or polarity isolation.

| Candidate | Sign accuracy | Median pref/null | Median DSI | Median polarity cross-talk |
|---|---:|---:|---:|---:|
| `(4,8,16)` | 0.108 | 2.553 | 0.437 | 1.059 |
| `(1,2,4,8)` | 0.122 | 4.261 | 0.620 | 1.145 |

No test split was used for this choice. No additional scale search was run.
