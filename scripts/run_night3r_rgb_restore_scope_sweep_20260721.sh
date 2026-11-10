#!/usr/bin/env bash
set -euo pipefail

cd /data/fcr/data/Night3R

PY="${PY:-/data/fcr/miniconda3/envs/stream3r/bin/python}"
OUTDIR="outputs/night3r_rgb_restore_scope_sweep_20260721"
CHECKPOINT="outputs/night3r_mvsec_only_adapt_1k_from_best/last_event_dark3r.pth"
DEPTH_CONFIG="config/train_night3r_mvsec_only_adapt_1k_from_best.yaml"
POSE_CONFIG="config/train_night3r_pose_separate_branches_w5_dense2k.yaml"
POSE_CHECKPOINT="outputs/night3r_pose_separate_branches_w5_dense2k_20260720/pose_heads/pose_head_step00001200.pth"
MAST3R_CHECKPOINT="checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"

mkdir -p "${OUTDIR}"

run_scope() {
  local gpu="$1" scope="$2"
  local scope_dir="${OUTDIR}/${scope}"
  mkdir -p "${scope_dir}"

  CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" utils/export_night3r_long_sequence.py \
    --checkpoint "${CHECKPOINT}" \
    --pose-head-checkpoint "${POSE_CHECKPOINT}" \
    --config "${POSE_CONFIG}" \
    --model-name-or-path "${MAST3R_CHECKPOINT}" \
    --restore-frozen-rgb-scope "${scope}" \
    --data-root /data/fcr/data/eval_data/DL3DV \
    --scene screen-000090 --start-frame 0 --num-frames 8 --image-size 512 \
    --exposure 0.10 --alignment gt_pose \
    --conf-quantile 0.75 --min-depth 0.0001 --max-depth 45 \
    --max-points-per-frame 18000 --voxel-size 0.01 \
    --amp --amp-dtype bf16 --device cuda --allow-unsafe-checkpoint-load \
    --outdir "${scope_dir}/dl3dv_gt_pose_8f" \
    > "${scope_dir}/dl3dv_gt_pose_8f.log" 2>&1

  CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" utils/eval_event_dark3r_variant.py \
    --variant event_dark3r \
    --checkpoint "${CHECKPOINT}" \
    --config "${DEPTH_CONFIG}" \
    --data-root /data/fcr/data/train_data/MVSEC_outdoor_night2_test \
    --model-name-or-path "${MAST3R_CHECKPOINT}" \
    --restore-frozen-rgb-scope "${scope}" \
    --allow-unsafe-checkpoint-load --event-mode real --no-lowlight-aug \
    --device cuda --amp --amp-dtype bf16 \
    --batch-size 1 --max-batches 32 \
    --pair-sampling fixed --pair-stride 1 \
    --min-pair-distance 1 --max-pair-distance 1 \
    --num-workers 0 --seed 17 \
    --output "${scope_dir}/mvsec_real_32.json" \
    > "${scope_dir}/mvsec_real_32.log" 2>&1
}

run_scope 4 head2 & p4=$!
run_scope 5 view2 & p5=$!
run_scope 6 heads & p6=$!
run_scope 7 geometry & p7=$!

status=0
for pid in "${p4}" "${p5}" "${p6}" "${p7}"; do
  wait "${pid}" || status=1
done

exit "${status}"
