# T4/T5-inspired cell validation

This is an engineering validation of separated ON/T4-inspired and OFF/T5-inspired
paths. It is not a biological-equivalence claim.

- Commit: `930c0c18fb56c3aa17edb56b34e123c26a89e37b`
- Seed: `20260823`
- Raster: `160x128` non-square, 5 frames
- Grid: 12 stimulus classes, 8 directions, 7 displacements, 3 contrasts,
  gamma/offset/noise/blur/border variants
- Checkpoint: none; fixed cell front-end
- Dataset manifest SHA256: `1cdb79d0c2c6dbec0c940424bf329ee97a552d1e410ed1cb6b4b11325690d37b`

## C0 result

The current `(4,8,16)` bank failed C0: direction sign accuracy `0.1078`,
median preferred/null `2.5533`, median DSI `0.4371`, and median polarity
cross-talk `1.0585`. Flicker and stationary false positives were low in this
implementation, but that does not compensate for the direction/polarity
failure.

One bounded small-motion scale correction `(1,2,4,8)` improved preferred/null
to `4.2613` and DSI to `0.6198`, but sign accuracy was only `0.1219` and
cross-talk worsened to `1.1448`. No further search was performed.

Machine-readable case results and representative panels are retained in the
operator scratch evidence for this commit; the compact public summary is
`t4t5_cell_validation.json`.
