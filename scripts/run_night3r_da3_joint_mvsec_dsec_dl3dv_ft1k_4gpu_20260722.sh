#!/usr/bin/env bash
set -euo pipefail

cd /data/fcr/data/Night3R

PY="${PY:-/data/fcr/miniconda3/envs/stream3r/bin/python}"
RUN_NAME="${RUN_NAME:-night3r_da3_joint_mvsec_dsec_dl3dv_ft1k_4gpu_20260722}"
SAVE_DIR="outputs/${RUN_NAME}"
CONFIG="${CONFIG:-config/train_night3r_da3_joint_mvsec_dsec_dl3dv_ft1k_20260722.yaml}"
INIT_CKPT="${INIT_CKPT:-outputs/night3r_rawrgb_nodarken_ft1k_from_final_4gpu_20260722/last_event_dark3r.pth}"
MAST3R_CKPT="${MAST3R_CKPT:-checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth}"
GPU_LIST="${GPU_LIST:-4,5,6,7}"
NPROC="${NPROC:-4}"
MASTER_PORT="${MASTER_PORT:-29731}"
MAX_STEPS="${MAX_STEPS:-1000}"
LOG="${SAVE_DIR}/train.log"

mkdir -p "${SAVE_DIR}"
exec > >(tee -a "${LOG}") 2>&1

echo "== Night3R joint MVSEC/DSEC/DL3DV-DA3 fine-tune =="
date -Is
echo "RUN_NAME=${RUN_NAME}"
echo "CONFIG=${CONFIG}"
echo "INIT_CKPT=${INIT_CKPT}"
echo "GPU_LIST=${GPU_LIST}"
echo "MAX_STEPS=${MAX_STEPS}"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits

CUDA_VISIBLE_DEVICES="${GPU_LIST}" \
"${PY}" -m torch.distributed.run \
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
  --gradient-checkpointing \
  --max-steps "${MAX_STEPS}" \
  --log-every 25

for step in 200 400 600 800 1000; do
  printf -v checkpoint "${SAVE_DIR}/event_dark3r_step%08d.pth" "${step}"
  test -f "${checkpoint}"
done
test -f "${SAVE_DIR}/last_event_dark3r.pth"
touch "${SAVE_DIR}/TRAIN_COMPLETE"
echo "== Training complete =="
date -Is
