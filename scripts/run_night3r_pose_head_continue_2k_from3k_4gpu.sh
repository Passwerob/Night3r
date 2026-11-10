#!/usr/bin/env bash
set -euo pipefail

cd /data/fcr/data/Night3R

PY="${PY:-/data/fcr/miniconda3/envs/stream3r/bin/python}"
GPU_LIST="${GPU_LIST:-4,5,6,7}"
NPROC="${NPROC:-4}"
MASTER_PORT="${MASTER_PORT:-29720}"
MAX_STEPS="${MAX_STEPS:-2000}"
RUN_POSE_EVAL="${RUN_POSE_EVAL:-1}"
RUN_NAME="${RUN_NAME:-night3r_pose_head_continue_2k_from3k_20260720}"
SAVE_DIR="outputs/${RUN_NAME}"
CONFIG="${CONFIG:-config/train_night3r_pose_head_continue_2k_from3k.yaml}"
INIT_CKPT="${INIT_CKPT:-outputs/night3r_pose_head_mvsec_dl3dv_3k_20260719/last_event_dark3r.pth}"
MAST3R_CKPT="${MAST3R_CKPT:-checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth}"

mkdir -p "${SAVE_DIR}"
exec > >(tee -a "${SAVE_DIR}/train_pose_head.log") 2>&1

echo "== Night3R pose-head continuation: 3K + 2K =="
date -Is
echo "GPU_LIST=${GPU_LIST} MAX_STEPS=${MAX_STEPS}"
echo "CONFIG=${CONFIG}"
echo "INIT_CKPT=${INIT_CKPT}"
test -f "${INIT_CKPT}"
test -f "${MAST3R_CKPT}"

CUDA_VISIBLE_DEVICES="${GPU_LIST}" "${PY}" -m torch.distributed.run \
  --nproc_per_node="${NPROC}" \
  --master_port="${MASTER_PORT}" \
  train_event_dark3r.py \
  --config "${CONFIG}" \
  --save-dir "${SAVE_DIR}" \
  --init-event-checkpoint "${INIT_CKPT}" \
  --model-name-or-path "${MAST3R_CKPT}" \
  --allow-unsafe-checkpoint-load \
  --device cuda \
  --amp \
  --amp-dtype bf16 \
  --max-steps "${MAX_STEPS}" \
  --log-every 25

date -Is
echo "Training complete: ${SAVE_DIR}/last_event_dark3r.pth"

if [[ "${RUN_POSE_EVAL}" == "1" ]]; then
  CHECKPOINT="${SAVE_DIR}/last_event_dark3r.pth" \
  CONFIG="${CONFIG}" \
  MAST3R_CKPT="${MAST3R_CKPT}" \
  GPU_LIST="${GPU_LIST}" \
  OUTPUT_DIR="${SAVE_DIR}/eval_direct_pose" \
    bash scripts/eval_night3r_direct_pose_controls_4gpu.sh
fi
