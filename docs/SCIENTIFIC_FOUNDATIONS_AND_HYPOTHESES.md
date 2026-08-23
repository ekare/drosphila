# Scientific foundations and hypotheses

This project uses the Drosophila circuit as an engineering inspiration and
does not claim biological equivalence. A literature result is not a project
acceptance result. The evidence classes are:

- `LITERATURE_EVIDENCE`: a primary paper or an explicitly labelled preprint.
- `PROJECT_EVIDENCE`: a reproducible test on this repository's data/code.
- `ENGINEERING_HYPOTHESIS`: a design choice awaiting a project test.
- `OPEN_QUESTION`: insufficient evidence or an unresolved blocker.

## Current hypothesis ledger

| ID | Hypothesis | Status | Evidence boundary |
|---|---|---|---|
| H1 | Preserve ON/OFF polarity before adaptation | supported | Deterministic phone golden: P2/P3 leakage 0/0.0017; real-texture field still fails. |
| H2 | Population code is better than exact argmax | inconclusive | Decoder exists; no complete argmax/population held-out comparison accepted. |
| H3 | Angular/ray scale transfers across raster/intrinsics | inconclusive | Angular conversion and arbitrary-intrinsics tests pass; learned field transfer fails. |
| H4 | D4 can match D8 | not_run | No controlled D4/D8 learned comparison completed. |
| H5 | Ray-aware local basis helps | not_run | Camera-ray contract exists; detector ablation not run. |
| H6 | LPi-like opponency improves robustness | not_run | Literature motivates it; no project ablation accepted. |
| H7 | LPTC templates improve global SO(3) | not_run | No accepted wide-field template experiment. |
| H8 | Competitive disinhibition separates translation common-mode | not_run | Translation gate is not available; no learned ablation. |
| H9 | Geometry-first SO(3) generalizes better | inconclusive | Oracle geometry primitives pass; learned local field fails before a fair comparison. |
| H10 | Longer causal context improves observability | not_supported_in_current_run | T3/T5/T9/T17 field audit did not pass; T17 had no observable cells. |
| H11 | Normal FOV is sufficient | inconclusive | 60/90/120 degree diagnostic all fail with current field; this identifies the current model as the blocker, not a proof of impossibility. |
| H12 | Latent angular memory can avoid pixel panorama | engineering_hypothesis | Oracle-safe angular feature memory exists and explicitly renders no pixels; no task-level retrieval gate. |
| H13 | Residual flow supports scale-free translation direction | oracle_only | Exact depth-conditioned residual geometry exists; no usable depth trajectory or learned RGB-only gate. |
| H14 | Topological memory should precede metric mapping | open_question | Scene-memory implementation and loop-closure metrics are not started. |

## Design constraints

Production input is one normal perspective phone camera with explicit
intrinsics, timestamps, and optional distortion/exposure/rolling-shutter
metadata. The project does not require a panorama and does not stitch pixels.
Motion is represented in camera-ray/angular coordinates where possible. Full
relative orientation is SO(3); a 1-D azimuth ring is only a visual projection.
Translation remains scale-free until an external scale source exists.

The registry and literature-to-architecture matrix are in
`references/primary_sources.yaml`, `references/phone_vision.bib`, and
`docs/LITERATURE_TO_PHONE_VISION_ARCHITECTURE.md`.
