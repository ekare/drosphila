# Translation and parallax oracle

`flyrot.geometry.translation_flow.rigid_camera_flow` now provides the exact
depth-conditioned rigid-flow primitive for an optical camera frame:

```text
X_next = R @ X_current + t
```

It returns full flow, pure rotational flow, the residual translation/parallax
flow, and a depth/visibility validity mask. `scale_free_translation_direction`
normalizes `t`; it intentionally does not claim metric scale.

This is an oracle geometry layer, not an RGB-only model input. A learned
translation release requires depth-verified data, an independent pose/depth
projection audit, trajectory-disjoint validation, and a held-out test. The
fast-cache index used by the current baseline has RGB and pose but no indexed
depth, so no translation model or metric-scale result is released here.
