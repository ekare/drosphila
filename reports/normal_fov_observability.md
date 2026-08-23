# Normal-FOV observability

status: **failed** as a diagnostic for the current field; not a proof that a
normal phone camera is impossible.

evidence: On the RTX 4060 Ti (`cuda:1`), the unchanged V1-A checkpoint was
evaluated at 256x144 with 24 samples per split for horizontal FOV 60°, 90°,
and 120°. Validation results:

| FOV | median angle | sign accuracy | magnitude Spearman | EPE improvement |
|---:|---:|---:|---:|---:|
| 60° | 92.63° | 0.453 | -0.202 | -5.00 |
| 90° | 89.09° | 0.488 | -0.276 | -5.61 |
| 120° | 86.18° | 0.508 | -0.340 | -5.10 |

The test split was reported after validation and was not used for selection.
Raw output is the operator-scratch artifact `phone_fov_observability.json`;
the public repository keeps only this portable summary.

next_required_condition: Do not switch production to a wide FOV. First repair
the field or replace the frontend, then repeat this diagnostic with the new
frozen candidate.
