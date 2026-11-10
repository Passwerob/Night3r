#!/usr/bin/env bash
set -euo pipefail

# Qualitative candidate screening on every strictly held-out MVSEC test scene.
# The resulting ranking is not a reported benchmark; it selects a curved scene
# before running all comparison methods on the final visualization candidate.

cd /data/fcr/data/Night3R

PY="${PY:-/data/fcr/miniconda3/envs/stream3r/bin/python}"
OUT="${OUT:-outputs/heldout_mvsec_trajectory_sweep_20260728/screen_sweep}"
DATA_ROOT="${DATA_ROOT:-/data/fcr/data/Night3R/test/MVSEC}"
CONFIG="${CONFIG:-config/train_night3r_final_step100_pose_adapt100.yaml}"
NIGHT3R_CKPT="${NIGHT3R_CKPT:-outputs/night3r_final_complete_pose_step100_20260726/event_dark3r_night3r_final_with_pose_head.pth}"
MAST3R_CKPT="${MAST3R_CKPT:-checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth}"
MAX_FRAMES="${MAX_FRAMES:-49}"
NUM_GPUS="${NUM_GPUS:-8}"

mkdir -p "${OUT}/json" "${OUT}/logs" "${OUT}/summary"
exec > >(tee -a "${OUT}/launcher.log") 2>&1

mapfile -t SCENES < <(find "${DATA_ROOT}" -mindepth 1 -maxdepth 1 -type d -name 'screen-*-mvsec' -printf '%f\n' | sort)
if [[ "${#SCENES[@]}" -eq 0 ]]; then
  echo "No MVSEC test scenes found in ${DATA_ROOT}" >&2
  exit 1
fi

run_scene() {
  local scene="$1" gpu="$2"
  local output="${OUT}/json/${scene}.json"
  [[ -s "${output}" ]] && return
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 "${PY}" \
    utils/eval_night3r_trajectory.py \
    --variant event_dark3r \
    --checkpoint "${NIGHT3R_CKPT}" \
    --config "${CONFIG}" \
    --data-root "${DATA_ROOT}" \
    --model-name-or-path "${MAST3R_CKPT}" \
    --allow-unsafe-checkpoint-load \
    --event-mode real --pose-recovery direct \
    --include-scenes "${scene}" --max-frames-per-scene "${MAX_FRAMES}" \
    --export-trajectories --num-workers 0 --device cuda --amp --amp-dtype bf16 \
    --output "${output}" \
    > "${OUT}/logs/${scene}.log" 2>&1
}

worker() {
  local gpu="$1" index
  for ((index = gpu; index < ${#SCENES[@]}; index += NUM_GPUS)); do
    echo "GPU ${gpu}: ${SCENES[index]}"
    run_scene "${SCENES[index]}" "${gpu}"
  done
}

pids=()
for ((gpu = 0; gpu < NUM_GPUS; gpu++)); do
  worker "${gpu}" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=1
done

"${PY}" utils/rank_heldout_mvsec_trajectory_candidates.py \
  --input-dir "${OUT}/json" \
  --output-dir "${OUT}/summary" || status=1

if [[ "${status}" -ne 0 ]]; then
  echo "One or more screening evaluations failed. Inspect ${OUT}/logs." >&2
  exit "${status}"
fi
touch "${OUT}/COMPLETE"
date -Is
