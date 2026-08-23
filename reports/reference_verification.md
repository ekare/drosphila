# Reference verification

status: **experimental**

- `references/primary_sources.yaml` contains 22 registry entries.
- 20 were checked against a primary article/preprint page during this run.
- 18 are peer-reviewed primary articles/conference papers; 2 are explicitly
  marked preprints (`FLIVVER` and the 2026 motion-based-depth paper).
- Official repositories were checked for FlyVis, DROID-SLAM, DPVO, FlyNet,
  and the motion-based-depth analysis. No external code is copied or imported.
- Two older article records remain `registry_reference_pending_direct_page_check`:
  REF-MOTION-04 and REF-NAV-01. They are not used as acceptance evidence.

The registry records repository refs as unpinned when no immutable commit was
needed for this architecture-only comparison. Pinning is required before any
external code becomes a runtime dependency.
