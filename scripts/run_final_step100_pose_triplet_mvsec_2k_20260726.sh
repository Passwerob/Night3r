#!/usr/bin/env bash
set -euo pipefail

cd /data/fcr/data/Night3R

PY="${PY:-/data/fcr/miniconda3/envs/stream3r/bin/python}"
GPU="${GPU:-4}"
MAX_STEPS="${MAX_STEPS:-2000}"
RUN_NAME="${RUN_NAME:-night3r_final_step100_pose_triplet_mvsec_2k_20260726}"
SAVE_DIR="outputs/${RUN_NAME}"
CONFIG="${CONFIG:-config/train_night3r_pose_triplet_mvsec_2k_20260726.yaml}"
BASE_CKPT="${BASE_CKPT:-outputs/night3r_final_step100_20260723/event_dark3r_night3r_final.pth}"
POSE_HEAD_CKPT="${POSE_HEAD_CKPT:-outputs/night3r_final_step100_pose_adapt100_20260726/pose_heads/pose_head_step00000100.pth}"
COMPOSED_CKPT="${COMPOSED_CKPT:-${SAVE_DIR}/initial_step100_with_pose_head.pth}"
MAST3R_CKPT="${MAST3R_CKPT:-checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth}"

mkdir -p "${SAVE_DIR}"
exec > >(tee -a "${SAVE_DIR}/run.log") 2>&1

test -f "${BASE_CKPT}"
test -f "${POSE_HEAD_CKPT}"
test -f "${MAST3R_CKPT}"

"${PY}" utils/compose_pose_head_checkpoint.py \
  --base-checkpoint "${BASE_CKPT}" \
  --pose-head-checkpoint "${POSE_HEAD_CKPT}" \
  --output "${COMPOSED_CKPT}" \
  --allow-unsafe-checkpoint-load

CUDA_VISIBLE_DEVICES="${GPU}" "${PY}" utils/train_night3r_pose_triplet.py \
  --config "${CONFIG}" \
  --save-dir "${SAVE_DIR}/pose_heads" \
  --init-event-checkpoint "${COMPOSED_CKPT}" \
  --model-name-or-path "${MAST3R_CKPT}" \
  --allow-unsafe-checkpoint-load \
  --device cuda --amp --amp-dtype bf16 \
  --max-steps "${MAX_STEPS}" --log-every 20

touch "${SAVE_DIR}/TRAIN_COMPLETE"
date -Is
