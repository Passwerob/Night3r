#!/usr/bin/env bash
set -euo pipefail

cd /data/fcr/data/Night3R

PY="${PY:-/data/fcr/miniconda3/envs/stream3r/bin/python}"
GPU_LIST="${GPU_LIST:-4,5,6,7}"
MASTER_PORT="${MASTER_PORT:-29721}"
MAX_STEPS="${MAX_STEPS:-800}"
RUN_NAME="${RUN_NAME:-night3r_descriptor_recovery_800_4gpu_20260721}"
SAVE_DIR="outputs/${RUN_NAME}"
CONFIG="config/train_night3r_descriptor_recovery_1k.yaml"
BASE_CHECKPOINT="outputs/night3r_mvsec_only_adapt_1k_from_best/last_event_dark3r.pth"
MAST3R_CHECKPOINT="checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
LOG="${SAVE_DIR}/train_and_sparse_sweep.log"

mkdir -p "${SAVE_DIR}"
exec > >(tee -a "${LOG}") 2>&1

echo "descriptor_recovery_start $(date -Is)"
echo "gpu_list=${GPU_LIST} max_steps=${MAX_STEPS} master_port=${MASTER_PORT}"

CUDA_VISIBLE_DEVICES="${GPU_LIST}" "${PY}" -m torch.distributed.run \
  --nproc_per_node=4 \
  --master_port="${MASTER_PORT}" \
  train_event_dark3r.py \
  --config "${CONFIG}" \
  --save-dir "${SAVE_DIR}" \
  --init-event-checkpoint "${BASE_CHECKPOINT}" \
  --model-name-or-path "${MAST3R_CHECKPOINT}" \
  --allow-unsafe-checkpoint-load \
  --device cuda \
  --amp \
  --amp-dtype bf16 \
  --image-size 128 128 \
  --max-steps "${MAX_STEPS}" \
  --log-every 20

echo "descriptor_recovery_train_done $(date -Is)"

STEPS=(200 400 600 800)
GPUS=(4 5 6 7)
PIDS=()
for INDEX in "${!STEPS[@]}"; do
  STEP="${STEPS[$INDEX]}"
  GPU="${GPUS[$INDEX]}"
  DESCRIPTOR_CHECKPOINT="${SAVE_DIR}/descriptor_head_step$(printf '%08d' "${STEP}").pth"
  EVAL_DIR="${SAVE_DIR}/sparse_ga_step$(printf '%04d' "${STEP}")"
  if [[ ! -f "${DESCRIPTOR_CHECKPOINT}" ]]; then
    echo "missing_descriptor_checkpoint ${DESCRIPTOR_CHECKPOINT}"
    exit 1
  fi
  CUDA_VISIBLE_DEVICES="${GPU}" "${PY}" \
    utils/export_night3r_sparse_global_sequence.py \
    --checkpoint "${BASE_CHECKPOINT}" \
    --descriptor-head-checkpoint "${DESCRIPTOR_CHECKPOINT}" \
    --config "${CONFIG}" \
    --model-name-or-path "${MAST3R_CHECKPOINT}" \
    --data-root /data/fcr/data/eval_data/DL3DV \
    --scene screen-000090 \
    --start-frame 0 \
    --num-frames 8 \
    --image-size 512 \
    --exposure 0.10 \
    --scene-graph swin-2-noncyclic \
    --init-focal 300 \
    --focal-reg-w 100 \
    --amp \
    --amp-dtype bf16 \
    --device cuda \
    --allow-unsafe-checkpoint-load \
    --outdir "${EVAL_DIR}" \
    > "${SAVE_DIR}/sparse_ga_step$(printf '%04d' "${STEP}").log" 2>&1 &
  PIDS+=("$!")
done

for PID in "${PIDS[@]}"; do
  wait "${PID}"
done

"${PY}" utils/summarize_descriptor_sparse_sweep.py \
  --run-dir "${SAVE_DIR}" \
  --output-dir "${SAVE_DIR}/sparse_ga_summary"

echo "descriptor_recovery_all_done $(date -Is)"
