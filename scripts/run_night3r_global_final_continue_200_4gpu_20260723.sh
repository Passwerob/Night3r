#!/usr/bin/env bash
set -euo pipefail

cd /data/fcr/data/Night3R

PY="${PY:-/data/fcr/miniconda3/envs/stream3r/bin/python}"
RUN_NAME="${RUN_NAME:-night3r_global_final_continue_200_4gpu_20260723}"
SAVE_DIR="outputs/${RUN_NAME}"
CONFIG="${CONFIG:-config/train_night3r_global_final_continue_200_20260723.yaml}"
INIT_CKPT="${INIT_CKPT:-outputs/night3r_three_dataset_global_final_20260723/event_dark3r_three_dataset_global_final.pth}"
MAST3R_CKPT="${MAST3R_CKPT:-checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth}"
GPU_CSV="${GPU_CSV:-4,5,6,7}"
MASTER_PORT="${MASTER_PORT:-29847}"
MAX_STEPS="${MAX_STEPS:-200}"

mkdir -p "${SAVE_DIR}"
exec > >(tee -a "${SAVE_DIR}/train.log") 2>&1

echo "== Night3R global-final continuation =="
date -Is
echo "CONFIG=${CONFIG}"
echo "INIT_CKPT=${INIT_CKPT}"
echo "GPU_CSV=${GPU_CSV}"
echo "MAX_STEPS=${MAX_STEPS}"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits

CUDA_VISIBLE_DEVICES="${GPU_CSV}" \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"${PY}" -m torch.distributed.run \
  --nproc_per_node=4 \
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
  --log-every 10

for step in 100 200; do
  printf -v checkpoint "%s/event_dark3r_step%08d.pth" "${SAVE_DIR}" "${step}"
  test -s "${checkpoint}"
done
test -s "${SAVE_DIR}/last_event_dark3r.pth"
touch "${SAVE_DIR}/TRAIN_COMPLETE"
echo "== Night3R global-final continuation complete =="
date -Is
