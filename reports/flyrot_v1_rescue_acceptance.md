# FlyRot-v1 rescue acceptance

Status: **implemented, experimental, not accepted**.

C0 failed with `cell_direction_failure` and `polarity_failure`. F0 then
failed with `local_field_failure`; the exact analytic solver itself passed its
geometry oracle but cannot repair an incorrect learned field. Therefore
G1-CONTROLLED and G1 real-sequence acceptance were not run as release gates.

FlyRot-v1 remains in the experimental branch/configuration, is not the default
model, does not replace the public v0.3 checkpoint, and has no v0.4.0 release.
