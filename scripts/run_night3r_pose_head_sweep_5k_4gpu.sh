#!/usr/bin/env bash
set -euo pipefail

cd /data/fcr/data/Night3R

PY="${PY:-/data/fcr/miniconda3/envs/stream3r/bin/python}"
GPU_LIST="${GPU_LIST:-4,5,6,7}"
NPROC="${NPROC:-4}"
MASTER_PORT="${MASTER_PORT:-29721}"
MAX_STEPS="${MAX_STEPS:-5000}"
RUN_NAME="${RUN_NAME:-night3r_pose_head_sweep_5k_20260720}"
SAVE_DIR="outputs/${RUN_NAME}"
HEAD_DIR="${SAVE_DIR}/pose_heads"
CONFIG="${CONFIG:-config/train_night3r_pose_head_sweep_5k.yaml}"
BASE_CKPT="${BASE_CKPT:-outputs/night3r_mvsec_only_adapt_1k_from_best/last_event_dark3r.pth}"
MAST3R_CKPT="${MAST3R_CKPT:-checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth}"

mkdir -p "${SAVE_DIR}" "${HEAD_DIR}"
exec > >(tee -a "${SAVE_DIR}/run.log") 2>&1

echo "== Fresh pose-head 5K sweep, checkpoints every 200 steps =="
date -Is
echo "GPU_LIST=${GPU_LIST} MAX_STEPS=${MAX_STEPS}"
echo "CONFIG=${CONFIG}"
echo "BASE_CKPT=${BASE_CKPT}"
test -f "${BASE_CKPT}"
test -f "${MAST3R_CKPT}"
bash scripts/prepare_pose_head_sweep_val_v1.sh

CUDA_VISIBLE_DEVICES="${GPU_LIST}" "${PY}" -m torch.distributed.run \
  --nproc_per_node="${NPROC}" \
  --master_port="${MASTER_PORT}" \
  train_event_dark3r.py \
  --config "${CONFIG}" \
  --save-dir "${HEAD_DIR}" \
  --init-event-checkpoint "${BASE_CKPT}" \
  --model-name-or-path "${MAST3R_CKPT}" \
  --allow-unsafe-checkpoint-load \
  --device cuda \
  --amp \
  --amp-dtype bf16 \
  --max-steps "${MAX_STEPS}" \
  --log-every 25

touch "${SAVE_DIR}/TRAIN_COMPLETE"
date -Is
echo "Training complete; starting validation sweep"

RUN_NAME="${RUN_NAME}" \
HEAD_DIR="${HEAD_DIR}" \
CONFIG="${CONFIG}" \
BASE_CKPT="${BASE_CKPT}" \
MAST3R_CKPT="${MAST3R_CKPT}" \
GPU_LIST="${GPU_LIST}" \
  bash scripts/eval_pose_head_sweep_val_4gpu.sh

touch "${SAVE_DIR}/VALIDATION_SWEEP_COMPLETE"
BEST_HEAD="$(${PY} -c "import json; print(json.load(open('${SAVE_DIR}/validation_sweep/selection.json'))['best_checkpoint'])")"
echo "Selected validation checkpoint: ${BEST_HEAD}"

CHECKPOINT="${BASE_CKPT}" \
POSE_HEAD_CHECKPOINT="${BEST_HEAD}" \
CONFIG="${CONFIG}" \
MAST3R_CKPT="${MAST3R_CKPT}" \
GPU_LIST="${GPU_LIST}" \
OUTPUT_DIR="${SAVE_DIR}/test_selected" \
  bash scripts/eval_night3r_direct_pose_controls_4gpu.sh

touch "${SAVE_DIR}/TEST_EVAL_COMPLETE"
date -Is
echo "Pose-head sweep complete"
