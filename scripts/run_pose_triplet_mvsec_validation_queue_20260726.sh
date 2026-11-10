#!/usr/bin/env bash
set -euo pipefail

cd /data/fcr/data/Night3R

PY="${PY:-/data/fcr/miniconda3/envs/stream3r/bin/python}"
GPU="${GPU:-5}"
RUN_NAME="${RUN_NAME:-night3r_final_step100_pose_triplet_mvsec_2k_20260726}"
RUN_DIR="outputs/${RUN_NAME}"
HEAD_DIR="${HEAD_DIR:-${RUN_DIR}/pose_heads}"
OUTPUT_DIR="${OUTPUT_DIR:-${RUN_DIR}/mvsec_validation_sweep}"
CONFIG="${CONFIG:-config/eval_night3r_pose_triplet_mvsec_validation_20260726.yaml}"
BASE_CKPT="${BASE_CKPT:-outputs/night3r_final_step100_20260723/event_dark3r_night3r_final.pth}"
MAST3R_CKPT="${MAST3R_CKPT:-checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth}"
DATA_ROOT="${DATA_ROOT:-/data/fcr/data/train_data/MVSEC}"
SCENES=("screen-000033-mvsec" "screen-000081-mvsec" "screen-000179-mvsec")
MAX_FRAMES="${MAX_FRAMES:-49}"
POLL_SECONDS="${POLL_SECONDS:-60}"

mkdir -p "${OUTPUT_DIR}"
exec > >(tee -a "${OUTPUT_DIR}/queue.log") 2>&1

eval_one() {
  local head="$1"
  local step
  step="$(basename "${head}" .pth | sed 's/pose_head_step//')"
  local step_dir="${OUTPUT_DIR}/step${step}"
  local output="${step_dir}/mvsec_validation.json"
  if [[ -s "${output}" ]]; then
    return
  fi
  mkdir -p "${step_dir}"
  echo "$(date -Is) evaluating step ${step}"
  CUDA_VISIBLE_DEVICES="${GPU}" "${PY}" utils/eval_night3r_trajectory.py \
    --variant event_dark3r \
    --checkpoint "${BASE_CKPT}" \
    --pose-head-checkpoint "${head}" \
    --config "${CONFIG}" \
    --data-root "${DATA_ROOT}" \
    --model-name-or-path "${MAST3R_CKPT}" \
    --allow-unsafe-checkpoint-load \
    --event-mode real --pose-recovery direct \
    --include-scenes "${SCENES[@]}" \
    --max-frames-per-scene "${MAX_FRAMES}" \
    --num-workers 0 --device cuda --amp --amp-dtype bf16 \
    --output "${output}" \
    > "${step_dir}/eval.log" 2>&1
}

while true; do
  mapfile -t heads < <(find "${HEAD_DIR}" -maxdepth 1 -type f -name 'pose_head_step*.pth' | sort)
  for head in "${heads[@]}"; do
    eval_one "${head}"
  done

  if [[ -f "${RUN_DIR}/TRAIN_COMPLETE" ]]; then
    pending=0
    for head in "${heads[@]}"; do
      step="$(basename "${head}" .pth | sed 's/pose_head_step//')"
      [[ -s "${OUTPUT_DIR}/step${step}/mvsec_validation.json" ]] || pending=1
    done
    if [[ "${pending}" == "0" && "${#heads[@]}" -gt 0 ]]; then
      break
    fi
  fi
  sleep "${POLL_SECONDS}"
done

touch "${OUTPUT_DIR}/COMPLETE"
date -Is
