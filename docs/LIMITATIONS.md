# Current limitations

- The released model estimates relative rotation only. It does not estimate
  translation, depth, optical flow, feature tracks, or a persistent world.
- The real-data directional residual is diagnostic, not a calibrated oracle:
  balanced real-data residuals for prediction, ground truth, and wrong sign are
  too close to establish correctness.
- The current TartanAir index used for the locked RGB+pose baseline has no
  indexed depth modality. Translation/parallax claims therefore require a
  separate depth-verified evaluation scope.
- The default TartanAir V2 intrinsics are documented as an explicit centered
  assumption until a camera calibration manifest is independently verified.
- The confidence output is a proxy and is not calibrated uncertainty.
- Orientation memory, translation, and scene memory remain experimental or
  unimplemented until their oracle and held-out quality gates are satisfied.
