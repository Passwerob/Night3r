#!/usr/bin/env bash
set -euo pipefail

cd /data/fcr/data/Night3R

PY="${PY:-/data/fcr/miniconda3/envs/stream3r/bin/python}"
RUN_DIR="outputs/night3r_long_gtpose_scale_fix_qsweep_dl3dv_8f_20260721"
GEOMETRY_CHECKPOINT="outputs/night3r_mvsec_only_adapt_1k_from_best/last_event_dark3r.pth"
POSE_CHECKPOINT="outputs/night3r_pose_separate_branches_w5_dense2k_20260720/pose_heads/pose_head_step00001200.pth"
CONFIG="config/train_night3r_pose_separate_branches_w5_dense2k.yaml"
MAST3R_CHECKPOINT="checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"

mkdir -p "${RUN_DIR}"
exec > >(tee -a "${RUN_DIR}/sweep.log") 2>&1

NAMES=(q065 q075 q085 q090)
GPUS=(4 5 6 7)
QUANTILES=(0.65 0.75 0.85 0.90)
PIDS=()

echo "gtpose_scale_fix_qsweep_start $(date -Is)"
for index in "${!NAMES[@]}"; do
  name="${NAMES[$index]}"
  gpu="${GPUS[$index]}"
  quantile="${QUANTILES[$index]}"
  outdir="${RUN_DIR}/${name}"
  echo "launch name=${name} gpu=${gpu} quantile=${quantile}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" \
    utils/export_night3r_long_sequence.py \
    --checkpoint "${GEOMETRY_CHECKPOINT}" \
    --pose-head-checkpoint "${POSE_CHECKPOINT}" \
    --config "${CONFIG}" \
    --model-name-or-path "${MAST3R_CHECKPOINT}" \
    --data-root /data/fcr/data/eval_data/DL3DV \
    --scene screen-000090 \
    --start-frame 0 \
    --num-frames 8 \
    --image-size 512 \
    --exposure 0.10 \
    --alignment gt_pose \
    --conf-quantile "${quantile}" \
    --min-depth 0.0001 \
    --max-depth 45 \
    --max-points-per-frame 18000 \
    --voxel-size 0.01 \
    --amp \
    --amp-dtype bf16 \
    --device cuda \
    --allow-unsafe-checkpoint-load \
    --outdir "${outdir}" \
    > "${RUN_DIR}/${name}.log" 2>&1 &
  PIDS+=("$!")
done

status=0
for index in "${!PIDS[@]}"; do
  if wait "${PIDS[$index]}"; then
    echo "complete name=${NAMES[$index]}"
  else
    echo "failed name=${NAMES[$index]}"
    status=1
  fi
done

echo "gtpose_scale_fix_qsweep_done status=${status} $(date -Is)"
exit "${status}"
