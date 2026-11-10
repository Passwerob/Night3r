#!/usr/bin/env bash
set -euo pipefail

cd /data/fcr/data/Night3R

PY="${PY:-/data/fcr/miniconda3/envs/stream3r/bin/python}"
VARIANT="${VARIANT:-shape}"
GPU_LIST="${GPU_LIST:-4,5}"
NPROC="${NPROC:-2}"
MASTER_PORT="${MASTER_PORT:-29781}"
MAX_STEPS="${MAX_STEPS:-800}"
INIT_CKPT="${INIT_CKPT:-outputs/night3r_final_tradeoff_head2_restored_20260723/event_dark3r_final_head2_restored.pth}"
MAST3R_CKPT="${MAST3R_CKPT:-checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth}"

case "${VARIANT}" in
  shape)
    RUN_NAME="${RUN_NAME:-night3r_da3_depth_repair_shape_800_2gpu_20260723}"
    CONFIG="${CONFIG:-config/train_night3r_da3_depth_repair_shape_800_20260723.yaml}"
    ;;
  metric)
    RUN_NAME="${RUN_NAME:-night3r_da3_depth_repair_metric_800_2gpu_20260723}"
    CONFIG="${CONFIG:-config/train_night3r_da3_depth_repair_metric_800_20260723.yaml}"
    ;;
  *)
    echo "Unknown VARIANT=${VARIANT}; expected shape or metric" >&2
    exit 2
    ;;
esac

SAVE_DIR="outputs/${RUN_NAME}"
LOG="${SAVE_DIR}/train.log"
mkdir -p "${SAVE_DIR}"
exec > >(tee -a "${LOG}") 2>&1

echo "== Night3R DA3 DL3DV depth repair: ${VARIANT} =="
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
  --log-every 20

for step in 100 200 300 400 500 600 700 800; do
  printf -v checkpoint "${SAVE_DIR}/event_dark3r_step%08d.pth" "${step}"
  test -f "${checkpoint}"
done
test -f "${SAVE_DIR}/last_event_dark3r.pth"
touch "${SAVE_DIR}/TRAIN_COMPLETE"
echo "== ${VARIANT} training complete =="
date -Is
