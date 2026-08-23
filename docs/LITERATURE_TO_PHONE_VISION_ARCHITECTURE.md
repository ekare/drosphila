# Literature to phone-vision architecture

The mapping below is deliberately conservative. Papers motivate a candidate
or an ablation; they do not license a biological-equivalence claim, and no
external repository is a production dependency.

| Literature concept | Project component | Decision | Evidence status |
|---|---|---|---|
| ON/OFF pathway separation; T4/T5 cardinal tuning | `PhonePolarityFrontend`, canonical direction bank | Adopt as abstract engineering contract | Project golden supports polarity preservation; field gate failed. |
| Offset excitation and delayed inhibition | bounded temporal/opponency candidates | Controlled ablation | Not run as a learned acceptance experiment. |
| LPi multilevel opponency | LPi-like local inhibition candidate | Controlled ablation | Not run. |
| LPTC wide-field optic-flow templates | global SO(3) readout candidate | Controlled ablation | Not run; current field fails first. |
| Competitive disinhibition | rotational-vs-common-mode branch | Controlled ablation | Blocked by missing accepted translation field. |
| FlyVis connectome-constrained model | teacher/reference | Reference only | Official repository inspected; no dependency or copied code. |
| Normal-perspective DROID-SLAM / DPVO | engineering baseline | Reference only | Supports feasibility boundary, not FlyRot correctness. |
| Ring-attractor heading models | `VisualAzimuthRing` | Adopt only as visual projection | Must not replace full SO(3). |
| FlyNet place recognition | future compact descriptor comparison | Reference only | Repository license has an academic/commercial-use condition. |
| FLIVVER and 2026 depth preprint | scale/depth hypotheses | Reference only | No metric scale or ready-made depth model is imported. |

The exact records, DOIs, verification state, repository refs, and license
notes are maintained in `references/primary_sources.yaml`. The two primary
repository implementations inspected for possible comparison are not copied
into this repository; their licenses remain their own.
