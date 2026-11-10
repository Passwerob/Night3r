#!/usr/bin/env bash
set -euo pipefail

cd /data/fcr/data/Night3R

PY="${PY:-/data/fcr/miniconda3/envs/stream3r/bin/python}"
RUN_DIR="outputs/night3r_long_pose_alignment_sweep_dl3dv_8f_20260721"
GEOMETRY_CHECKPOINT="outputs/night3r_mvsec_only_adapt_1k_from_best/last_event_dark3r.pth"
MAST3R_CHECKPOINT="checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
SEPARATE_POSE="outputs/night3r_pose_separate_branches_w5_dense2k_20260720/pose_heads/pose_head_step00001200.pth"
SEPARATE_CONFIG="config/train_night3r_pose_separate_branches_w5_dense2k.yaml"
ORIGINAL_POSE="outputs/night3r_pose_head_sweep_5k_20260720/pose_heads/pose_head_step00001600.pth"
ORIGINAL_CONFIG="config/train_night3r_pose_head_sweep_5k.yaml"

mkdir -p "${RUN_DIR}"
exec > >(tee -a "${RUN_DIR}/sweep.log") 2>&1

NAMES=(pose_separate1200 pose_original1600 shared_frame_sim3 gt_pose_upper_bound)
GPUS=(4 5 6 7)
ALIGNMENTS=(pose_head pose_head shared_frame_sim3 gt_pose)
POSE_CHECKPOINTS=("${SEPARATE_POSE}" "${ORIGINAL_POSE}" "${SEPARATE_POSE}" "${SEPARATE_POSE}")
CONFIGS=("${SEPARATE_CONFIG}" "${ORIGINAL_CONFIG}" "${SEPARATE_CONFIG}" "${SEPARATE_CONFIG}")
PIDS=()

echo "long_pose_alignment_sweep_start $(date -Is)"
for index in "${!NAMES[@]}"; do
  name="${NAMES[$index]}"
  gpu="${GPUS[$index]}"
  alignment="${ALIGNMENTS[$index]}"
  pose_checkpoint="${POSE_CHECKPOINTS[$index]}"
  config="${CONFIGS[$index]}"
  outdir="${RUN_DIR}/${name}"
  echo "launch name=${name} gpu=${gpu} alignment=${alignment}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" \
    utils/export_night3r_long_sequence.py \
    --checkpoint "${GEOMETRY_CHECKPOINT}" \
    --pose-head-checkpoint "${pose_checkpoint}" \
    --config "${config}" \
    --model-name-or-path "${MAST3R_CHECKPOINT}" \
    --data-root /data/fcr/data/eval_data/DL3DV \
    --scene screen-000090 \
    --start-frame 0 \
    --num-frames 8 \
    --image-size 512 \
    --exposure 0.10 \
    --alignment "${alignment}" \
    --conf-quantile 0.75 \
    --min-depth 0.25 \
    --max-depth 45 \
    --max-points-per-frame 18000 \
    --voxel-size 0.08 \
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

echo "long_pose_alignment_sweep_done status=${status} $(date -Is)"
exit "${status}"
