# C0/F0 before and after correctness audit

Historical reports are preserved. Before is the v0.3.1 experimental
evaluator; after is a no-training rerun with the explicit image-plane
conversion and separate polarity/magnitude diagnostics.

| Gate | Before | After | Interpretation |
|---|---:|---:|---|
| C0 direction contract | partially invalid | direction cells pass canonical population direction | Old channel/plane interpretation was wrong |
| C0 ON/OFF separation | not isolated by pure edge | failed under P0; P2 control passed | Frontend cross-talk remains |
| C0 argmax | primary historical metric | secondary only | Adjacent ties require tie-aware/population metrics |
| F0 T3 direction | 89.14 deg | 75.93 deg | Improved after contract correction, still failed |
| F0 T5 direction | 89.74 deg | 74.53 deg | Improved after contract correction, still failed |
| F0 T7 direction | 95.20 deg | 74.82 deg | Improved after contract correction, still failed |
| F0 magnitude | negative Spearman | unavailable/un-calibrated; rho 0.464 on deterministic tuning | No px decoder is justified |
| F0 EPE | negative improvement | negative improvement | Corrected decoder does not make V1 field pass |

The final diagnosis is: validation_contract_bug_confirmed,
direction_cells_work, polarity_frontend_failed, and magnitude_decoder_failed.
The next architecture work requires a repaired or gated polarity frontend and
a separately calibrated magnitude readout.
