# Continuous Pose Trajectory Evaluation

## Purpose

This evaluation tests whether Night3R can compose pairwise relative poses into
a usable continuous camera trajectory under low-light RGB-event input. It is a
trajectory and temporal-geometry diagnostic, not a claim of non-rigid dynamic
object reconstruction.

## Frozen Model Composition

- Geometry checkpoint: `outputs/night3r_final_step100_20260723/event_dark3r_night3r_final.pth`.
- Pose initialization: the separate-branch pose head selected at step 1200 by
  validation in `night3r_pose_separate_branches_w5_dense2k_20260720`.
- Adaptation: the pose head alone was fine-tuned for 100 steps at learning rate
  `1e-4`; all Night3R geometry, event encoder, fusion, and geometry-head
  parameters had zero learning rate.
- Training sources: MVSEC and DL3DV samples with relative-pose labels. The
  inherited training exclusion list removes all paper MVSEC/DL3DV test scenes.

## Protocol

For every adjacent pair in a 49-frame sequence, Night3R directly predicts
`T_camera(t+1)_from_camera(t)` with its pose head. Poses are composed as
`T_world_camera(t+1) = T_world_camera(t) inv(T_camera(t+1)_from_camera(t))`.
MASt3R does not expose a direct pose head; its pair pose is recovered from
bidirectional dense point maps by confidence-filtered trimmed Sim(3)
registration. Both methods use every adjacent edge and one global Sim(3)
alignment to the GT trajectory for translation evaluation. No per-frame
alignment is applied.

ATE, RPE-t, and RPE-R are reported after this single global alignment. The
figure's second panel plots per-frame translation error after the same
alignment. All trajectories contain 49 frames and 48 adjacent pairs.

## Results

| Dataset / sequence | Input | Method | ATE (m) | RPE-t (m) | RPE-R (deg) |
|---|---|---|---:|---:|---:|
| MVSEC `screen-000157-mvsec` | Stored low-light RGB + real events | Night3R | 0.281 | 0.128 | 0.992 |
| MVSEC `screen-000157-mvsec` | Stored low-light RGB | MASt3R | 0.900 | 0.367 | 5.203 |
| Held-out DL3DV `screen-000088` | 10% linear-domain exposure + noise | Night3R | 0.457 | 0.164 | 1.710 |
| Held-out DL3DV `screen-000088` | 10% linear-domain exposure + noise | MASt3R | 0.968 | 0.268 | 1.930 |

The rendered figures and raw trajectories are under:

`outputs/dynamic_pose_trajectory_step100_20260726/figures/`

## Reporting Scope

- DL3DV `screen-000088` is held out from the Night3R DL3DV training root and
  is suitable as a strict qualitative trajectory example.
- The MVSEC scene is an adaptation-domain diagnostic. It must not be labeled
  as a strict unseen-domain result because the repository's MVSEC evaluation
  inventory overlaps the broader MVSEC source used by some adaptation stages.
- This is a one-sequence visual comparison per dataset. It supports the
  continuous-reconstruction figure, while a paper-wide pose table should
  aggregate all predefined sequences using the same protocol.
- MASt3R and Night3R do not use the same pose-output mechanism: Night3R uses
  its trained direct pose head, while MASt3R uses point-map registration. The
  figure therefore compares each method's available end-to-end pose recovery
  route and should state this in the appendix protocol.

## Long-Range MVSEC Diagnostic

For an internal long-range diagnostic, nine consecutive `outdoor_night2`
exports (`screen-000153-mvsec` through `screen-000161-mvsec`) were assembled
without copying RGB or events. The virtual sequence contains 441 frames and a
130.2 m GT path. Exported chunk boundaries introduce up to a 50.1 ms temporal
gap, so this is a sparse long-range diagnostic rather than the standard
adjacent-frame test protocol.

Night3R obtains 100% pair coverage, `RPE-t = 0.181 m`, and
`RPE-R = 0.510 deg`, but the accumulated long-range `ATE` is `10.010 m` and the
rotation APE RMSE is `71.042 deg`. The resulting figure visibly shows the
accumulated drift over the large turn in the GT path. It is useful for debugging
long-horizon pose accumulation, but it must not be used as evidence that
Night3R currently solves long-range metric trajectory reconstruction.

Long-range outputs:

`outputs/dynamic_pose_trajectory_long_mvsec_20260726/figures/night3r_vs_gt/`
