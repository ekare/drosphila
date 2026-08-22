# FlyRot roadmap

This roadmap separates verified capabilities from research targets. A target
does not become a release claim until it has a dataset-specific evaluator,
provenance, held-out validation, and the GTX 1050 Ti inference check.

1. Lock the v0.3.0 RGB-only rotation baseline and portable provenance.
2. Make pose conventions, camera intrinsics, crop/resize transforms, and
   image-motion geometry executable contracts with synthetic golden tests.
3. Add trajectory- and axis-aware evaluation with explicit invalidity reasons.
4. Develop FlyRot-v1 as a separate experimental model with explicit ON/OFF
   paths, retinotopic evidence, and per-step SO(3) composition.
5. Evaluate orientation memory, scale-free translation, and scene memory as
   separate gates. Keep oracle-only results distinct from learned results.
6. Add export and hardware reports only after a model passes the preceding
   quality gates.
