# T4/T5 validation correctness audit

This audit is evaluator-only. It does not modify v0.3.1, model weights, the
source dataset, or the historical C0/F0 reports.

## Contract result

The requested image-plane convention is now explicit:

- x right is positive and y down is positive.
- Angles start at east and increase clockwise: E, SE, S, SW, W, NW, N, NE.
- The historical channel order is converted explicitly with
  [0, 7, 6, 5, 4, 3, 2, 1].

Treating the old channel order as canonical produces a D4 flip_y solution
with 0 degree population-vector error. After the explicit conversion, the
best mapping is identity with 0 degree error. This confirms a validation
contract bug, not a weight change.

## C0 and polarity

The corrected deterministic direction cells decode the pure edge direction,
but the active P0 frontend leaks approximately 0.833 of the matched response
into the opposite polarity on both ON and OFF edges. P1 is similar at 0.822.
The fixed P2 variant suppresses this leakage to zero on this audit stimulus.
Bars, patches, gratings, and contrast reversals were kept out of pure-edge
cross-talk metrics because they contain mixed polarity events.

Argmax is not the primary decision because the response population contains
adjacent ties. The audit reports argmax, tie-aware, circular-neighbor, signed
angular, and population-vector cosine metrics separately.

## F0 and magnitude

The V1-A checkpoint was evaluated with the corrected canonical decoder on
T3/T5/T7, with train, validation, and test enumerated without test-based
selection. Validation direction remains failed: approximately 75.9, 74.5,
and 74.8 degrees median angular error for T3, T5, and T7. EPE improvement
versus zero is negative for all three.

The deterministic response curve over 0.5, 1, 2, 4, 8, 12, 16 displacement
units has Spearman monotonicity 0.464 and a second-peak/peak ratio of 0.918.
It is therefore reported as unavailable_un_calibrated; no physical-pixel
claim is made.

Raw JSON, traces, stimulus strips, mapping matrices, field quiver, commands,
and hashes are in the NV1 scratch bundle named
t4t5_validation_correctness_evidence.tar.gz.
