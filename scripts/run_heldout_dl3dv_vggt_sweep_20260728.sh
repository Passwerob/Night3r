#!/usr/bin/env bash
set -euo pipefail

cd /data/fcr/data/Night3R

PY="${PY:-/data/fcr/miniconda3/envs/stream3r/bin/python}"
OUT="${OUT:-outputs/heldout_dl3dv_trajectory_shortlist_20260728/vggt_sweep_fixed}"
DATA_ROOT="${DATA_ROOT:-/data/fcr/data/Night3R/test/DL3DV}"
CONFIG="${CONFIG:-config/train_night3r_final_step100_pose_adapt100.yaml}"
VGGT_CKPT="${VGGT_CKPT:-/data/fcr/ckpt/model.pt}"
SCENES=(
  screen-000127
  screen-000108
  screen-000088
  screen-000112
  screen-000087
  screen-000094
  screen-000090
  screen-000093
)

mkdir -p "${OUT}/inputs" "${OUT}/json" "${OUT}/logs"
exec > >(tee -a "${OUT}/launcher.log") 2>&1

run_scene() {
  local scene="$1" gpu="$2"
  local output="${OUT}/json/${scene}.json"
  local isolated_root="${OUT}/inputs/${scene}"
  [[ -s "${output}" ]] && return
  mkdir -p "${isolated_root}"
  ln -sfn "${DATA_ROOT}/${scene}" "${isolated_root}/${scene}"
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 "${PY}" \
    utils/eval_vggt_trajectory.py \
    --variant vggt --checkpoint "${VGGT_CKPT}" --base-vggt-checkpoint "${VGGT_CKPT}" \
    --config "${CONFIG}" --data-root "${isolated_root}" --max-scenes 1 --max-frames-per-scene 49 \
    --lowlight-exposure 0.10 --export-trajectories --num-workers 0 --device cuda --amp --amp-dtype bf16 \
    --output "${output}" > "${OUT}/logs/${scene}.log" 2>&1
}

pids=()
for index in "${!SCENES[@]}"; do
  run_scene "${SCENES[index]}" "${index}" &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=1
done
if [[ "${status}" -ne 0 ]]; then
  echo "One or more VGGT evaluations failed. Inspect ${OUT}/logs." >&2
  exit "${status}"
fi
touch "${OUT}/COMPLETE"
date -Is
