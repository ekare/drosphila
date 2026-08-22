# FlyRot v0.3.0 architecture

FlyRot-v0 is intentionally small. It is a biologically inspired motion
front-end followed by a geometry-aware evidence readout.

```text
RGB sequence
    |
    v
adaptive luminance photoreceptor
    |
    +--> ON temporal changes
    +--> OFF temporal changes
              |
              v
8 direction-selective local correlation cells
at spatial scales (4, 8, 16)
              |
              v
camera-intrinsics-aware rotational evidence
using exact perspective geometry for diagnostics
              |
              v
causal leaky evidence accumulator
              |
              v
linear SO(3) rotation readout
              |
              +--> relative rotation vector
              +--> magnitude-based confidence proxy
              +--> validity-aware directional diagnostics
```

## Photoreceptor stage

RGB is converted to luminance, locally normalized, and differenced over time.
Positive differences form the ON channel and negative differences form the OFF
channel. This removes much of the static appearance and emphasizes motion.

## Direction-selective stage

Each cell compares delayed and current ON/OFF responses after a spatial shift.
The response is evaluated in eight compass directions and at three spatial
scales. The implementation uses valid masks at image boundaries instead of
wrapping pixels around the image.

## Geometry stage

Local directional energy is projected onto the three basis fields produced by
camera rotation. The diagnostic API accepts full pinhole intrinsics `(fx, fy,
cx, cy, width, height)` and also preserves the centered legacy focal-ratio
form. Exact perspective rotation geometry is used for residual comparison; a
finite-difference rotational basis supplies observability and least-squares
evidence.

The primary residual is `directional_residual_ratio`: unexplained native
direction evidence divided by observed native energy. It is deliberately not
called a physical displacement ratio. `residual_ratio` is retained only as a
deprecated compatibility alias.

## Temporal state

The causal accumulator maintains a leaky state. At each step, new motion
evidence is mixed with a learned fraction of the previous state. It never uses
future frames, so the design remains compatible with streaming inference.

## Output and limitations

The readout predicts a three-component rotation vector. It does not explicitly
estimate translation, depth, object identity, or a persistent world map. A
translating camera can produce image motion that resembles rotation when scene
depth is unknown; this is the main current research limitation.

Temporal, scale, ON/OFF, observability, and global-reliability fields expose
their own validity masks. An unavailable optional score is not converted to
zero. Global reliability is a validity-aware aggregation and is deliberately
documented as an abstention heuristic, not calibrated uncertainty.
