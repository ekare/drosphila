# FlyRot-v0 architecture

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
rotational-flow basis projection
using normalized camera coordinates
              |
              v
causal leaky evidence accumulator
              |
              v
linear SO(3) rotation readout
              |
              +--> relative rotation vector
              +--> magnitude-based confidence proxy
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
camera rotation. Image coordinates are normalized using the camera intrinsics
convention supplied by the model configuration. This provides a shared
geometric explanation for local motion rather than treating each pixel as an
independent classifier.

## Temporal state

The causal accumulator maintains a leaky state. At each step, new motion
evidence is mixed with a learned fraction of the previous state. It never uses
future frames, so the design remains compatible with streaming inference.

## Output and limitations

The readout predicts a three-component rotation vector. It does not explicitly
estimate translation, depth, object identity, or a persistent world map. A
translating camera can produce image motion that resembles rotation when scene
depth is unknown; this is the main current research limitation.

The confidence proxy is derived from predicted rotation magnitude and fixed
training-distribution constants. It is deliberately documented as an
abstention heuristic, not as calibrated uncertainty.
