# Orientation memory

FlyRot-v1 exposes two different quantities:

- `step_rotation_vector`: the per-pair estimate;
- `rotation_matrix` and `rotation_vector`: the cumulative ordered SO(3)
  composition of those steps.

Composition is matrix multiplication in the declared optical frame, followed
by an SO(3) logarithm for the cumulative vector. It is not a component-wise
sum of Euler angles. The implementation has a deterministic non-commuting
two-step test.

The current v1 candidate is experimental. Its held-out test was slightly worse
than the locked v0.3 baseline and its real-data directional oracle remained
unresolved, so orientation memory is implemented but not release-validated.
