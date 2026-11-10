# Ablation Section Organization

## Required Evidence Chain

The ablation section should answer four different questions in this order.
They should not be presented as one flat list of module removals.

1. **Do synchronized events provide information?**
   Use the Real/Zero/Shuffle control.  It is the causal attribution test.
2. **Why use BRG for event fusion?**
   Use fusion alternatives for the quantitative trade-off and a true gate-map
   visualization for mechanism inspection.
3. **How is heterogeneous supervision made trainable?**
   Use the loss table to identify metric depth as the dominant geometric
   constraint and active-loss normalization/source weighting as training
   stabilizers.
4. **Why is a direct pose head needed for sequential reconstruction?**
   Treat this as a reconstruction-output capability, not as another BRG
   ablation.  The direct head replaces unreliable pose recovery from pairwise
   point maps while all geometry modules remain frozen.

## Main-Paper Layout

### 1. Aligned Event Measurements

Keep `table3_event_input_control.tex` in the main paper.  It is the strongest
evidence that the model uses the event stream: the network, RGB input, pairs,
and GT mask are fixed, while only the event tensor changes.

Safe claim:

> Real events outperform Zero and Shuffle on the reported low-light MVSEC and
> DSEC controls.  The gain therefore requires aligned event structure, not
> merely extra parameters or the marginal event-value distribution.

The normal-light `zurich_city_02_b` result can be mentioned in the supplement
as a **single-scene case study**.  It must not be pooled with the low-light
benchmark table or described as a general DSEC daylight average.

### 2. BRG: Quantitative Fusion Trade-off and Qualitative Gate Evidence

Keep only the **fusion-operator** block of `table1_fusion_BRG.tex` in the
main paper, preferably condensed to Full BRG, scalar cross-attention,
Concat--MLP residual, and additive projection.  Direct RGB replacement can be
one appendix row or a sentence because it tests a different, destructive
output mode.

The current numerical evidence supports:

- BRG is substantially safer than scalar cross-attention and additive
  projection under the matched protocol.
- Replacing RGB tokens is not cross-domain stable.
- Concat--MLP is numerically tied with, or slightly ahead of, BRG on the
  displayed metrics.  BRG must therefore not be claimed to be universally or
  numerically superior to every learned fusion operator.

Use the BRG figure in the main paper, but create a **new compact actual-gate
panel**.  Its minimum content should be:

| Low-light RGB | signed event activity | deployed post-occupancy BRG gate | gated-event residual weight |
|---|---|---|---|

Use one MVSEC and one DSEC example with the same gate color scale within each
example.  The plotted gate must be the inference-time `g_i` *after* the hard
occupancy mask, upsampled only for display.  If a residual-weight map cannot
be exported exactly, use the post-mask gate map alone and label it accordingly.

Do **not** use the existing
`brg_stat_sensitivity_display_only.png` as the main BRG figure.  It recomputes
pre-occupancy display-only gates after masking individual statistics; it is
useful supplementary material but does not show the deployed gate and cannot
be used to claim that each statistic has a large geometric effect.

Suggested main-paper caption:

> **BRG event modulation.** For synchronized low-light RGB-event inputs, the
> deployed BRG assigns a bounded token-wise event coefficient.  The occupancy
> mask suppresses injection where no event support is present; the map is the
> actual post-mask inference gate, shown only after token-grid upsampling for
> visualization.

### 3. Training Objective

Keep `table4_dynamic_loss_supervision.tex` in the main paper, because the
remaining losses are central to the final method.  Organize rows by objective:

- loss aggregation: active-loss normalization, source weights, static loss;
- privileged teacher supervision: point-map consistency;
- metric geometry: metric-depth loss.

Safe claim:

> Metric-depth supervision is the dominant scale-sensitive signal.  Point-map
> supervision and dynamic aggregation are complementary; their role is stable
> multi-source optimization, not a replacement for metric depth.

Do not reintroduce removed photo, depth-gradient, edge, or descriptor losses
in the text or table.  The broad paragraph about event-window length, voxel
bin count, hot pixels, burst events, and quality filtering should move to the
supplement unless a compact result table is actually shown.

### 4. Pose-Aware Sequential Reconstruction

Make this a standalone subsection after the pairwise geometry results and
before generic ablations, for example:

```tex
\subsection{Pose-Aware Sequential Reconstruction}
\label{subsec:pose}
```

The direct pose head is not a generic accuracy tweak.  It produces the
relative transform used to chain pairwise reconstructions, while the depth and
point-map heads remain unchanged.  Its input is the final cross-view decoder
tokens, not decoded depth, point maps, or descriptors.

Use the existing paper-ready table:

`outputs/pose_head_ablation_mvsec_full_20260727/table_pose_head_ablation.tex`

It compares the same frozen Night3R geometry with and without the direct head
on 558 adjacent edges from 18 pose-head-held-out MVSEC chunks:

| Variant | ATE | RPE-t | RPE-r | Pair rotation | Valid edges |
|---|---:|---:|---:|---:|---:|
| Direct pose head | 0.0932 | 0.1101 | 0.4925 deg | 0.3763 deg | 558/558 |
| Point-map Sim(3), strict | -- | -- | 45.0587 deg | 45.0587 deg | 24/558 |
| Point-map Sim(3), relaxed diagnostic | 0.4680 | 0.3785 | 52.1659 deg | 51.9466 deg | 558/558 |

Use this interpretation:

> Pairwise point maps provide strong depth but are not sufficiently
> cross-view-consistent for reliable relative-pose recovery.  A pose head
> trained on frozen decoder representations makes every evaluated edge
> available for trajectory composition; accepting weak geometry-only
> registrations yields large translation and rotation drift.

Important limits:

- ATE and translation RPE use one Sim(3) alignment per scene.  They measure
  trajectory shape/drift, not absolute physical-scale pose recovery.
- The direct head's night-chunk errors are higher than its non-night errors,
  but the strict geometry-only fallback fails on both groups.  The current
  experiment supports the need for a direct pose output under the mixed
  RGB-event protocol; it does **not** isolate a claim that only low light
  causes point-map pose failure.
- The 18 chunks are held out from pose-head adaptation.  Do not imply that the
  frozen geometry checkpoint was never exposed to the broader MVSEC source
  root.

Add one 3D trajectory figure only if it uses a fixed disclosed held-out night
chunk and compares Direct head / GT / relaxed point-map fallback.  It should
support the table rather than replace it.  The existing multi-method trajectory
figures are suitable for the broader sequential-reconstruction comparison, not
for the pose-head causal ablation.

## Appendix-Only Material

Move the following from the main Ablation Studies subsection to the appendix:

- individual density, occupancy, and polarity-statistic removal table;
- source-bias and source-statistic-normalization table;
- no-gate-regularization and legacy-channel-gate table;
- adapter-scope table;
- BRG pre-occupancy display-only statistic-sensitivity maps;
- low-light construction variants and event-noise robustness details.

They are useful for reproducibility and mechanism audit, but their measured
effects are small or they add too much structure for the main evidence chain.

## Minimal Main Text Skeleton

```tex
\subsection{Pose-Aware Sequential Reconstruction}
% pose table + one concise paragraph

\subsection{Ablation Studies}
\textbf{Aligned event measurements.}
% Table 3 + causal attribution paragraph

\textbf{Reliability-aware fusion.}
% compact fusion table + actual post-mask BRG gate figure

\textbf{Heterogeneous supervision.}
% Table 4 + loss interpretation paragraph
```

This layout gives every retained experiment one job: event controls establish
causality, fusion and gate maps justify the design, the loss table justifies
training, and the pose table justifies sequential reconstruction.

