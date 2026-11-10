# Night3R

Night3R is the cleaned event-based 3D reconstruction branch extracted from the
Dark3R working tree. The actual model lineage is MASt3R-first: a MASt3R
backbone is wrapped with an event token encoder and event/RGB fusion adapter.
Dark3R LoRA weights are kept only as an optional low-light RGB backbone
initialization.

## Release

Source code is in this repository. The paper-final checkpoints are on Hugging Face, both without and with the adapted pose head:

- [Passwerob/Night3R](https://huggingface.co/Passwerob/Night3R)
  - `event_dark3r_night3r_final.pth` — geometry checkpoint (global-final continue, step 100)
  - `event_dark3r_night3r_final_with_pose_head.pth` — the same geometry plus the adapted pose head
  - `pose_head_step00000100.pth` — pose head only

The consolidated MVSEC / DSEC / DL3DV evaluation scenes are at [Passwerob/Night3R-test](https://huggingface.co/datasets/Passwerob/Night3R-test).

Loading still needs the public MASt3R backbone `MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth`. That third-party checkpoint is not redistributed in this release.

## What Is Kept

- Event training/export entry points:
  - `train_night3r.py`
  - `forward_night3r.py`
- Historical compatibility entry points:
  - `train_event_dark3r.py`
  - `forward_event_dark3r.py`
- Core code:
  - `model/`
  - `dataset/`
  - `losses/`
  - `training/`
  - `utils/`
  - `mast3r/`
  - `dust3r/`
- Final weights only:
  - base MASt3R checkpoint and retrieval weights
  - optional Dark3R LoRA `0049_lora.pth`
  - `last_event_dark3r.pth` from each retained experiment

Intermediate epoch/stage checkpoints, previous `outputs/` visualizations,
rendered videos, logs, caches, Dark3R-NeRF, raw download utilities, and old RGB
Dark3R training/inference entry points were intentionally not copied.

## Model Strategy

Night3R uses:

```text
MASt3R backbone
  + event voxel encoder
  + token-level event density statistics
  + optional identity-residual Density-FiLM
  + scalar/cross-attention/tensor/BRG event fusion
  + MASt3R decoder/head outputs
```

The event density path computes token-level statistics from raw event voxels:

```text
[log event density, occupancy, polarity balance]
```

Those statistics can condition Density-FiLM and tensor-gated event fusion. The
main implementation still uses historical names such as `EventDark3RStudent`,
`event_dark3r`, and `last_event_dark3r.pth` because existing checkpoints were
created with those module/config names.

The current default ablation target is `Night3RBoundedEventFusion` / BRG:

```text
fused = rgb_tokens_raw + event_scale * g_token * g_channel * W_e(LN(event_tokens))
```

BRG keeps the MASt3R token stream as the residual base, uses a token reliability
gate `[B,N,1]` from normalized event statistics, bounds the learnable event
scale, and defaults Density-FiLM off. Legacy tensor/simple fusion remains
selectable through config for regression tests.

## Weights

Important retained weights:

```text
checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth
checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl
checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth
checkpoints/dark3r_pretrained/weights/0049_lora.pth
checkpoints/*/last_event_dark3r.pth
checkpoints/from_outputs/outputs/*/last_event_dark3r.pth
```

`checkpoints/from_outputs/` contains final weights and small metadata copied
from previous output directories. It does not contain intermediate checkpoints.

## Training

Example:

```bash
cd /data/fcr/data/Night3R
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  train_night3r.py \
  --config config/train_night3r_brg_token_reg_bf16.yaml \
  --save-dir outputs/night3r_brg_token_reg \
  --device cuda \
  --allow-unsafe-checkpoint-load
```

The existing configs still use `train_event_dark3r_*` names for compatibility.
Absolute checkpoint paths have been rewritten to `/data/fcr/data/Night3R`.

Legacy tensor no-FiLM control:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  train_night3r.py \
  --config config/train_night3r_no_film_tensor_gate_bf16.yaml \
  --save-dir outputs/night3r_legacy_tensor_no_film \
  --device cuda \
  --allow-unsafe-checkpoint-load
```

Additional fusion ablation overrides are listed in
`config/night3r_ablation_overrides.md`.

## Export Pair Predictions

```bash
python forward_night3r.py \
  --config config/train_event_dark3r_mvsec_dsec.yaml \
  --checkpoint checkpoints/from_outputs/outputs/event_dark3r_mvsec_dsec_train/last_event_dark3r.pth \
  --data-root /path/to/screen_dataset_root \
  --outdir outputs/export_example \
  --device cuda \
  --allow-unsafe-checkpoint-load
```

## Data Format

The dataset loader expects `screen-*` scene directories:

```text
screen-XXXX/
  images/*.png
  events/*.pt
  depth_gt/*.pt              optional
  cameras.yaml               optional
  transforms.json            optional
  meta/poses_selected.npz    optional
  meta.json                  optional
```

Each sample returns two RGB frames, two event voxels, intrinsics, optional
depth, and optional relative pose.

## Paper Table Metric Protocol

The LaTeX tables under `tables/` were audited on 2026-07-16 using only existing
evaluation JSON files. No experiment was rerun to update these tables.

Two depth protocols are stored by the evaluator and must not be mixed:

```text
depth_raw_absrel
depth_median_aligned_absrel
```

Raw AbsRel preserves the predicted metric scale. Median-aligned AbsRel rescales
each predicted pair by the valid-pixel median ratio before computing the error.
Earlier table drafts labeled raw values only as `AbsRel`; this made several
MVSEC ablations appear as `0.4+` even though their median-aligned values are
`0.2+`. Every table now names the protocol in its column header.

Table data sources and conventions:

- `table0_main_results.tex` reports both raw and median-aligned AbsRel from
  `outputs/exp1_follow_20260711_128/main_results_summary.csv`. Night3R is the
  final MVSEC-only 1k checkpoint: MVSEC raw/median AbsRel is
  `0.2073/0.1949`, and DSEC is `0.1356/0.0559`.
- `table1_fusion_BRG.tex` reports median-aligned AbsRel. The completed two-GPU
  fusion runs used the 612-pair DSEC holdout, while the later one-GPU fusion
  runs used 128 DSEC-current pairs. These are separate columns and must not be
  compared as if they were one test split.
- `table2_event_statistics_source_calibration.tex`,
  `table4_dynamic_loss_supervision.tex`, `table5_dataset_composition.tex`, and
  `table6_low_light_training.tex` report median-aligned AbsRel from the
  corresponding `eval_after_train/*_real_128.json` files.
- `table3_event_input_control.tex` intentionally reports both raw and
  median-aligned metrics for Real/Zero/Shuffle using the final MVSEC-only 1k
  checkpoint. Its source is `outputs/final_mvsec1k_event_controls_20260715/`.
- `table7_extreme_low_light.tex` reports median-aligned AbsRel from the already
  completed mixed-6k checkpoint at exposures 0.05 and 0.10. It is a low-light
  diagnostic and is not the final-1k event-input table.

The `\best{}` marker is assigned from the displayed metric within a compatible
evaluation block. It is not carried over from raw values after a table is
converted to median alignment. A dash means that exact checkpoint/split was not
evaluated; values are never copied across DSEC-holdout and DSEC-current.
