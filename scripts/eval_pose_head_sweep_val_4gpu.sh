#!/usr/bin/env bash
set -euo pipefail

cd /data/fcr/data/Night3R

PY="${PY:-/data/fcr/miniconda3/envs/stream3r/bin/python}"
RUN_NAME="${RUN_NAME:-night3r_pose_head_sweep_5k_20260720}"
RUN_DIR="outputs/${RUN_NAME}"
HEAD_DIR="${HEAD_DIR:-${RUN_DIR}/pose_heads}"
OUTPUT_DIR="${OUTPUT_DIR:-${RUN_DIR}/validation_sweep}"
CONFIG="${CONFIG:-config/train_night3r_pose_head_sweep_5k.yaml}"
BASE_CKPT="${BASE_CKPT:-outputs/night3r_mvsec_only_adapt_1k_from_best/last_event_dark3r.pth}"
MAST3R_CKPT="${MAST3R_CKPT:-checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth}"
MVSEC_VAL="${MVSEC_VAL:-outputs/splits/pose_head_sweep_val_v1/MVSEC}"
DL3DV_VAL="${DL3DV_VAL:-outputs/splits/pose_head_sweep_val_v1/DL3DV}"
GPU_LIST="${GPU_LIST:-4,5,6,7}"

IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
mapfile -t HEADS < <(find "${HEAD_DIR}" -maxdepth 1 -type f -name 'pose_head_step*.pth' | sort)
if [[ "${#HEADS[@]}" -eq 0 ]]; then
  echo "No pose-head checkpoints found in ${HEAD_DIR}" >&2
  exit 2
fi
if [[ "${#GPUS[@]}" -eq 0 ]]; then
  echo "GPU_LIST is empty" >&2
  exit 2
fi
mkdir -p "${OUTPUT_DIR}"

eval_one() {
  local gpu="$1"
  local head="$2"
  local dataset="$3"
  local root="$4"
  local step
  step="$(basename "${head}" .pth | sed 's/pose_head_step//')"
  local step_dir="${OUTPUT_DIR}/step${step}"
  local output="${step_dir}/${dataset}_real.json"
  mkdir -p "${step_dir}"
  if [[ -s "${output}" ]]; then
    return
  fi
  local exposure_args=()
  if [[ "${dataset}" == "dl3dv" ]]; then
    exposure_args+=(--lowlight-exposure 0.05)
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" utils/eval_night3r_trajectory.py \
    --variant event_dark3r \
    --checkpoint "${BASE_CKPT}" \
    --pose-head-checkpoint "${head}" \
    --config "${CONFIG}" \
    --data-root "${root}" \
    --model-name-or-path "${MAST3R_CKPT}" \
    --allow-unsafe-checkpoint-load \
    --event-mode real \
    --pose-recovery direct \
    --max-scenes 3 \
    --max-frames-per-scene 32 \
    --device cuda \
    --amp \
    --amp-dtype bf16 \
    --num-workers 0 \
    "${exposure_args[@]}" \
    --output "${output}" \
    > "${output%.json}.stdout.log" 2>&1
}

worker() {
  local worker_index="$1"
  local gpu="${GPUS[worker_index]}"
  local index
  for index in "${!HEADS[@]}"; do
    if (( index % ${#GPUS[@]} != worker_index )); then
      continue
    fi
    eval_one "${gpu}" "${HEADS[index]}" mvsec "${MVSEC_VAL}"
    eval_one "${gpu}" "${HEADS[index]}" dl3dv "${DL3DV_VAL}"
  done
}

pids=()
for worker_index in "${!GPUS[@]}"; do
  worker "${worker_index}" > "${OUTPUT_DIR}/worker${worker_index}.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  wait "${pid}"
done

"${PY}" utils/summarize_pose_head_sweep.py \
  --eval-root "${OUTPUT_DIR}" \
  --output-dir "${OUTPUT_DIR}"
