# Drosphila Phone-Vision master start state

This is the portable start-state record for the master development branch. It
intentionally omits usernames, hostnames, IP addresses, absolute local paths,
GPU UUIDs, and private storage identifiers.

## Repository and release boundary

- Development branch: codex/phone-vision-master
- Starting commit: 191327773fa657f6b6170a5de04e9f81930cca18
- Stable origin/main: 970400abafa1273ea1674f73223a5a6e3c055d5b
- Stable v0.3.1 tag target: 970400abafa1273ea1674f73223a5a6e3c055d5b
- v0.4.0 tag: absent
- Stable release: v0.3.1; it contains no new model checkpoint
- Working tree at branch creation: clean

The master branch was created from the correctness-audit branch so the
validated canonical image-plane conversion, evaluator, tests, and failure
evidence remain available. This does not change stable main or any release.

## Pull requests

| PR | State | Scope |
|---:|---|---|
| 2 | open | historical mixed experimental foundation |
| 3 | merged | v0.3.1 stable correctness and provenance |
| 4 | open | experimental T4/T5-inspired field and FlyRot-v1 |
| 5 | open | T4/T5 validation correctness audit |

## Verified software and hardware classes

- Python: 3.11
- PyTorch: 2.5.1 with CUDA 12.1 runtime
- available GPU classes: GTX 1050 Ti and RTX 4060 Ti
- validated audit device: CUDA ordinal 1, RTX 4060 Ti
- persistent operator scratch: NV1 verified; public report omits its local path

## Existing evidence boundary

The correctness audit is the authoritative interpretation of the historical
T4/T5 and F0 evaluator. It found the historical channel/plane convention
partially invalid, active P0 polarity leakage, failed corrected V1-A field
direction, and unavailable physical magnitude calibration. Translation and
scene memory remain oracle-only or unstarted capabilities.

## Start-state policy

The next work begins with the normal-phone camera and ray-space contract.
Normal perspective RGB is the production input. Panorama and pixel stitching
are diagnostic exclusions, not production dependencies. All later gates must
retain split provenance, exact commands, configuration/checkpoint hashes, and
explicit evidence classes.
