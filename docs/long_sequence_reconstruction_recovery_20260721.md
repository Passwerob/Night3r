# Night3R Long-Sequence Reconstruction Recovery

Date: 2026-07-21

## Scope

This audit addresses the fan-shaped long-sequence point clouds produced by the
final depth-selected Night3R checkpoint:

`outputs/night3r_mvsec_only_adapt_1k_from_best/last_event_dark3r.pth`

The depth-selected checkpoint remains the model used for the reported depth
tables. The recovery modes below are evaluated separately and do not overwrite
that checkpoint or its existing results.

## Root Cause

The final checkpoint differs substantially from the pristine MASt3R geometry
path even though the intended method freezes that path. Relative L2 parameter
drift is approximately 15.8% for `downstream_head1`, 16.8% for
`downstream_head2`, and 1.9-2.1% for the two decoder branches. The RGB encoder
drift is much smaller, approximately 0.095%.

The final checkpoint gives strong first-view depth, but its two point-map
outputs no longer share a reliable rigid coordinate system. Consequently,
pairwise confidence filtering, correspondence filtering, GT pose chaining, and
per-pair scale estimation cannot repair the original fan-shaped reconstruction.

## Implemented Recovery Modes

The evaluators now support selective restoration without creating another
multi-gigabyte full checkpoint:

```text
--restore-frozen-rgb-scope {none,head2,view2,heads,geometry,all}
--restore-frozen-rgb-alpha ALPHA
--geometry-head-checkpoint CHECKPOINT
```

`scope=heads` targets both MASt3R downstream heads. `alpha=0` keeps the final
Night3R heads, while `alpha=1` uses the pristine MASt3R heads. Intermediate
values linearly interpolate only these head parameters. Event encoder, BRG,
fusion adapter, RGB encoder, cross-view decoder, and pose head remain unchanged.

## Main Findings

### Selective restoration

- Restoring only `head2` or the complete second-view branch does not work. The
  first shared-frame scale remains negative and the fused cloud remains a fan.
- Restoring both downstream heads restores positive shared-frame scales and
  recognizable geometry.
- Restoring the complete RGB path also restores geometry, but degrades direct
  depth too severely.

### Head interpolation sweep

| Head alpha | MVSEC Raw AbsRel | MVSEC Raw delta1 | 8-frame reconstruction |
|---:|---:|---:|:---|
| 0.00 | 0.3513 | 0.6900 | fan-shaped |
| 0.25 | 0.3198 | 0.7093 | fan-shaped |
| 0.50 | 0.3056 | 0.7107 | fragmented |
| 0.75 | 0.3028 | 0.6981 | recognizable, visible double surfaces |
| 0.90 | 0.3049 | 0.6852 | stable and recognizable |
| 1.00 | 0.3075 | 0.6743 | stable and recognizable |

These numbers use the same first 32 MVSEC test pairs. The alpha sweep therefore
identifies `alpha=0.90` as the qualitative reconstruction setting: it is above
the geometric stability transition while retaining slightly more of the
depth-adapted head than full restoration.

### Full 128-pair depth check

| Dataset | Setting | Raw AbsRel | Raw delta1 | Median AbsRel | Median delta1 |
|:---|:---|---:|---:|---:|---:|
| MVSEC | final depth checkpoint | 0.2073 | 0.8118 | 0.1949 | 0.8016 |
| MVSEC | heads, alpha=0.90 | 0.2224 | 0.6648 | 0.2121 | 0.7614 |
| DSEC | final depth checkpoint | 0.1356 | 0.8373 | 0.0559 | 0.9826 |
| DSEC | heads, alpha=0.90 | 0.2143 | 0.4132 | 0.0835 | 0.9434 |

The blended-head setting must not replace the final model in the depth tables.
It is an inference-time geometry recovery mode for qualitative long-sequence
fusion. A single model that jointly preserves the final depth metrics and
cross-view point-map rigidity still requires retraining with explicit
bidirectional point-map supervision.

## 16-Frame Estimated-Pose Results

Both runs use 10% exposure, the separately trained pose head, 15 adjacent
pairs, confidence quantile 0.75, and `heads/alpha=0.90`.

| DL3DV scene | Frames | Fused voxels | Estimated path length | Assessment |
|:---|---:|---:|---:|:---|
| screen-000090 | 16 | 148,594 | 1.5024 | recognizable planter, ground, and vegetation |
| screen-000083 | 16 | 115,481 | 0.7346 | coherent wall/sign plane; weaker teaser viewpoint |

Primary qualitative output:

`outputs/night3r_heads_blend090_final_validation_20260721/screen-000090_pose_head_16f/night3r_long_reconstruction_teaser.png`

Secondary scene:

`outputs/night3r_heads_blend090_final_validation_20260721/screen-000083_pose_head_16f/night3r_long_reconstruction_teaser.png`

## Short Geometry-Head Training Attempt

An 800-step, four-GPU recovery run was also completed. Only the two downstream
heads were trainable; all event modules, BRG, RGB encoder, decoder, and pose head
were frozen. Training used symmetric pairs, clean-RGB teacher point maps, and
available MVSEC/DSEC depth.

The run remained numerically stable and did not OOM, but all saved nodes
degraded MVSEC depth. Step 400 was the least poor tested node with Raw AbsRel
0.4300 on the same 32-pair subset. These trained checkpoints are retained for
audit purposes but are not selected.

Training output:

`outputs/night3r_geometry_head_recovery_800_4gpu_20260721`

## Recommended Use

1. Use the unchanged final checkpoint for all reported DSEC/MVSEC depth metrics.
2. Use `scope=heads, alpha=0.90` only for current qualitative long-sequence
   reconstruction figures and demos.
3. Do not report the blended-head depth metrics as the main Night3R result.
4. For a unified future checkpoint, start from pristine MASt3R heads and train
   with symmetric input order, teacher point-map supervision on both outputs,
   and an explicit shared-frame rigidity/valid-edge criterion for checkpoint
   selection.

## Reproduction Scripts

- `scripts/run_night3r_restored_rgb_validation_20260721.sh`
- `scripts/run_night3r_rgb_restore_scope_sweep_20260721.sh`
- `scripts/run_night3r_heads_restore_final_check_20260721.sh`
- `scripts/run_night3r_geometry_head_recovery_800_4gpu.sh`
- `scripts/run_night3r_heads_blend_sweep_20260721.sh`
- `scripts/run_night3r_heads_blend090_final_validation_20260721.sh`
