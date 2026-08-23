# Polarity frontend selection

Evidence class: PROJECT_EVIDENCE. This is a bounded deterministic golden
benchmark, not a biological-equivalence result and not a learned-model
acceptance gate.

## Candidates

- P0: historical per-frame local log-luminance normalization followed by
  temporal difference.
- P1: log-luminance temporal difference with local common-mode subtraction.
- P2: temporal difference followed by sign-preserving divisive adaptation.
- P3: P2 plus bounded frame common-mode subtraction.

## Result

The benchmark used canonical pure ON/OFF translating edges on 320x180,
256x144, and 192x108 rasters with off-center, non-square intrinsics. The
exact command and full rows are stored in the operator scratch evidence
manifest named phone_golden_benchmark.json.

| Candidate | ON to OFF leakage | OFF to ON leakage | Ideal direction median | Gate |
|---|---:|---:|---:|---|
| P0 | 0.4623 | 0.4623 | 0.047 deg | failed |
| P1 | 0.5276 | 0.5276 | 0.079 deg | failed |
| P2 | 0.0000 | 0.0000 | 0.000 deg | passed |
| P3 | 0.0017 | 0.0017 | 0.000 deg | passed |

Flicker and stationary response ratios were zero for all candidates in this
bounded bank measurement. P2 is the provisional selected frontend because it
passes with the smallest mechanism and no learned parameters. P3 remains a
controlled common-mode-inhibition ablation; it is not silently merged into
the production model.

## Boundary

This supports H1 on the tested golden stimuli: the historical P0 normalization
can corrupt polarity separation. It does not establish real-texture
generalization, motion-field acceptance, or rotation acceptance. Perturbed
edges and real-texture validation remain required before downstream training.
