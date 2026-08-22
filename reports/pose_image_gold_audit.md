# Pose/image-motion and intrinsics contract audit

## Synthetic golden gate

Status: **PASS**.

- Exact SO(3) homography projection agrees with `exact_rotational_flow` for
  non-square, off-center intrinsics to absolute tolerance `1e-10` in the
  golden test.
- Correct-sign flow has positive median dot product with the homography
  displacement; the wrong-sign flow has negative median dot product.
- Crop and resize propagate `fx, fy, cx, cy` with the explicit pixel-center
  rule.
- Named pose targets are tested for `R_t.T @ R_t+1`, image-motion transpose,
  and NED-to-optical basis change.

## Real RGB+pose sanity check

Status: **PASS as a directional sanity check only; not a quality gate**.

The check used 32 indexed `House`, `Office`, `ModernCityDowntown`, and
`Cyberpunk` trajectory-camera records, four pairs per record, gap 4, and the
documented focal assumption 320 px. All 128 pairs produced LK scores.

| candidate | wins | median cosine |
| --- | ---: | ---: |
| `R_rel.T` | 72 / 128 | 0.07682 |
| `R_rel` | 56 / 128 | -0.03839 |

The weak cosine and mixed wins are expected under translation, depth, moving
content, feature-tracking failures, and an assumed focal length. This result
supports keeping the transpose convention explicit; it does not validate a
pure-rotation oracle or a learned model.

## Intrinsics sensitivity

The active v0.3 compatibility path at 128x128 uses centered `cx=cy=63.5`.
Resizing the tracked TartanAir V2 assumption `f=320, cx=cy=320` from 640x640
with the pixel-center rule gives `cx=cy=63.6`. For rotation vector
`[0.12,-0.08,0.05]`, the resulting exact-flow difference is at most `0.068`
pixel and mean `0.017` pixel. This is small for the baseline but is recorded
as an assumption boundary rather than silently folded into the release claim.

## Depth boundary

The fast-cache index used for the baseline contains RGB and pose only: zero
indexed depth records. Depth directories exist in a separate local source
copy for some trajectories, but they were not read or used in this audit and
are not redistributed. Translation/parallax acceptance remains unresolved.
