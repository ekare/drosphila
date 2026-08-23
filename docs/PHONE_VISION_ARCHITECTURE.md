# Phone-vision architecture

The production boundary is a single normal perspective phone camera. The
input contract carries `frames[B,T,3,H,W]`, per-frame intrinsics, timestamps,
and optional distortion, exposure, rolling-shutter, and crop/resize metadata.

```text
phone RGB + K + timing
        |
        v
camera rays / angular displacement
        |
        v
polarity-preserving ON/OFF frontend
        |
        v
retinotopic direction populations
        |
        v
local motion field --(accepted field required)--> full relative SO(3)
                                                   |
                                                   v
                         rotational compensation -> residual parallax
                                                   |
                                                   v
                                      scale-free translation direction
                                                   |
                                                   v
                              compact topological keyframe memory
```

The diagram is a gated architecture, not a claim that every downstream block
is currently accepted. The present evidence accepts the camera/ray contract,
bounded polarity candidates, and oracle geometry only. The learned local field
fails its gate, so the SO(3), translation, path-integration, and scene-memory
blocks remain oracle-only or not started.

No panorama is required. No pixel stitching is used. A visual azimuth ring is
only a projection for a compact angular state; full orientation remains SO(3).
Metric translation scale is not inferred without an external scale source.
