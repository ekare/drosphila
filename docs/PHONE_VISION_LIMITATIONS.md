# Phone-vision limitations

Current project evidence does not support a phone-camera rotation release.

- The historical image-plane evaluator had a convention bug; the explicit
  canonical conversion fixes that evaluator interpretation.
- The active P0 polarity frontend fails pure-edge cross-talk.
- P2 passes the bounded deterministic polarity gate, but it has not been
  trained or validated as part of V1-A.
- V1-A local motion fields fail corrected real-texture phone-raster direction
  and EPE gates.
- Magnitude is unavailable without calibration.
- Translation is oracle-only/unstarted.
- Orientation memory, angular feature memory, scene memory, and mobile export
  are not accepted capabilities.
- A 60/90/120 degree normal-FOV diagnostic also failed with the unchanged
  field; this does not prove that normal FOV is impossible, only that FOV is
  not the sole current explanation.

The project uses Drosophila-inspired, T4/T5-inspired, LPi-like, and
LPTC-like language only. It does not claim biological equivalence, metric SLAM,
full fly-brain simulation, or full world understanding.
