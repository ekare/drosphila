# Drosophila phone-vision master report

status: **experimental**

This is the single handover for the continuous master task. Stable `main` and
v0.3.1 remain untouched; development continues on the dedicated master
branch. The attached task document is treated as the roadmap and gate
contract, while every result below is classified by its actual evidence.

## Accepted boundaries

- Single normal perspective phone input with explicit camera/ray metadata.
- No production panorama and no pixel stitching.
- Camera geometry, crop/resize, distortion, timing, and angular conversion
  tests pass.
- P2 passes the bounded deterministic polarity gate; P3 remains an ablation.
- Exact SO(3), depth-conditioned rigid flow, rotational compensation, and
  scale-free translation helpers are oracle-safe and tested.
- Orientation and angular feature memories are implemented as oracle-only
  primitives; angular memory reports that it does not render pixels.

## Failed or incomplete gates

- Historical P0/P1 polarity leakage failed; the unchanged V1-A field fails
  phone-raster direction/sign/EPE gates.
- T17 has no observable cells in the bounded field run.
- 60°, 90°, and 120° FOV diagnostics all fail with the current field.
- Learned SO(3), learned translation, path integration, scene memory, and
  streaming/export are not accepted.
- No metric scale is claimed.

## Next autonomous work

Repair or replace the local motion field, rerun the frozen phone contracts and
controlled/real gates, then execute the downstream translation and memory
gates. Runtime/export checks are performed only for a capability-gated API.

Machine-readable status is in `drosphila_phone_vision_master_summary.json`.
