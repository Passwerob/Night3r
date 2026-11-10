# Normal-Light Event Ablation Plan

## Question and claim boundary

This experiment asks a narrow question: when the RGB frames are already
well-exposed, does the deployed Night3R event pathway still provide useful
geometry, rather than degrading the RGB reconstruction?

It must **not** be presented as evidence that events universally dominate RGB
in ordinary illumination.  The supported claim is limited to native daytime
driving scenes with synchronized events and valid metric depth.

## Selected Reporting Protocol: strict held-out DSEC daylight

The primary benchmark is formed from two strict DSEC scene-level holdouts:

| Split | Source sequence | Frames | Adjacent pairs | Status |
|---|---|---:|---:|---|
| `dsec_003` | `zurich_city_02_b` | 613 | 612 | **Selected normal-light result** |
| `dsec_004` | `zurich_city_11_b` | 967 | 966 | Internal diagnostic only |
| `dsec_extended` | Both above | 1580 | 1578 | Internal diagnostic only |

`screen-000002-dsec` is excluded because it overlaps DSEC training.  The
archived `screen-000005-dsec` is also excluded.  The reported normal-light
claim uses only `screen-000003-dsec` (`zurich_city_02_b`), which contains
native daytime RGB frames, real synchronized events, and sparse metric depth
from DSEC disparity.  The other strict scene and pooled result are retained to
audit the conclusion boundary, but are not reported as the normal-light main
result.  Since this case was selected after the multi-scene diagnostic was
available, it is a single-scene case study rather than a dataset-average or a
claim of universal normal-light benefit.

Each evaluation uses fixed stride-one adjacent pairs, the native input RGB
(`e=1.0`), no synthetic darkening, and no synthetic shot or read noise.  The
final complete Night3R checkpoint, pair order, validity masks, and evaluator
are fixed for all conditions.

## Event-input controls

The main evidence is a same-checkpoint intervention.  Only `event_voxel` is
changed; RGB, GT depth, model parameters, and pair order are identical.

| Condition | Intervention | What it tests |
|---|---|---|
| Real | Original synchronized event voxel | Full deployed model |
| Zero | Replace the voxel by zeros | Whether the event residual itself helps beyond the retained RGB path |
| Shuffle | Deterministic spatial and channel roll of the voxel | Whether any gain requires RGB-event alignment rather than event magnitude or extra inputs |

The evaluator's zero condition makes the BRG occupancy mask zero, so the
event residual is exactly disabled while the RGB pathway remains intact.
Shuffle preserves tensor shape and event-value distribution, but breaks the
spatial correspondence and temporal/polarity channel correspondence.

For these same-model controls, report both raw and per-pair median-aligned
AbsRel and delta1.  Raw values diagnose whether event input changes the model's
metric scale; aligned values test relative geometry independently of scale.
The primary comparison is `Real` versus `Zero`; `Real` versus `Shuffle` is the
alignment control.  State the signed differences as
`gain_AbsRel = AbsRel(control) - AbsRel(real)` and
`gain_delta1 = delta1(real) - delta1(control)`, so a positive value means that
real events help.

## Fusion-specific normal-light ablation

The event-input controls establish that the final event branch is useful.  A
separate matched 10K training set of existing checkpoints evaluates whether
the reliability-aware residual formulation is preferable to simple fusion even
when the RGB images are clean:

| Variant | Checkpoint |
|---|---|
| Full BRG | `outputs/exp3_full_brg_1gpu_10k_20260717/last_event_dark3r.pth` |
| Concat + MLP residual | `outputs/exp6_fusion_concat_mlp_1gpu_10k_20260717/last_event_dark3r.pth` |
| Additive projection | `outputs/exp6_fusion_additive_1gpu_10k_20260717/last_event_dark3r.pth` |

All three were trained from the same pre-adaptation initialization under the
same 10K cap.  They are evaluated with real events on `dsec_003` at native RGB
exposure.  This is an architecture ablation, not a replacement for the final
checkpoint control: do not mix their absolute numbers with final-model
results.

## Required reporting

1. Report `dsec_003` as the normal-light result.  Keep `dsec_004` and the
   1578-pair pool as internal diagnostics; do not use them to broaden the
   normal-light claim.
2. Use identical valid-depth masks and per-pair median alignment for every
   compared condition.
3. Include the event-gate mean and residual-ratio diagnostics in the output.
   They are explanatory diagnostics, not quality metrics.
4. Do not select pairs, scenes, or thresholds after inspecting the Real--Zero
   result.  The scene list and stride are fixed above.
5. If Real is only tied with Zero on the full DSEC pool, report this as
   non-degradation; do not claim a normal-light improvement.  A positive claim
   requires a consistent Real advantage over both controls.

## Execution

Run `scripts/run_normal_light_event_ablation_20260728.sh` inside tmux with four
available GPUs.  The script creates an isolated symlink root containing only
the declared scenes, runs the three final-model controls and the three matched
fusion variants, then writes machine-readable JSON/CSV/Markdown summaries to
`outputs/normal_light_event_ablation_20260728/`.
