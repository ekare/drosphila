# PR #2 and v0.3.1 release scope

## Decision

PR #2 remains open because its original branch mixes stable correctness and
provenance work with experimental FlyRot-v1, T4/T5-inspired polarity and
retinotopic-field work, SO(3) orientation memory, and translation-oracle
helpers. CI is green. The reason it remains open is therefore a release-scope
gate, not an authentication or permission failure.

Stable work was split to `codex/v0.3.1-correctness` and merged as PR #3 at
`970400abafa1273ea1674f73223a5a6e3c055d5b`. Package version is `0.3.1`; tag
and release `v0.3.1` exist. The existing v0.3 checkpoint and v0.3.0 tag were
not changed. The release contains no new learned weights.

The experimental line is `codex/experimental-t4t5-v1`, tracked in PR #4, and
is not default behavior. No `v0.4.0` tag or release was created.
