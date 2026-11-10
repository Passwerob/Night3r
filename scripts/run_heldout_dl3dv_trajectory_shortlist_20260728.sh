#!/usr/bin/env bash
set -euo pipefail

# Screen the curved, strictly held-out DL3DV trajectory candidates under the
# same deterministic 10% low-light protocol used for the final comparison.

cd /data/fcr/data/Night3R

PY="${PY:-/data/fcr/miniconda3/envs/stream3r/bin/python}"
OUT="${OUT:-outputs/heldout_dl3dv_trajectory_shortlist_20260728}"
DATA_ROOT="${DATA_ROOT:-/data/fcr/data/Night3R/test/DL3DV}"
CONFIG="${CONFIG:-config/train_night3r_final_step100_pose_adapt100.yaml}"
NIGHT3R_CKPT="${NIGHT3R_CKPT:-outputs/night3r_final_complete_pose_step100_20260726/event_dark3r_night3r_final_with_pose_head.pth}"
MAST3R_CKPT="${MAST3R_CKPT:-checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth}"
EXPOSURE="${EXPOSURE:-0.10}"
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

mkdir -p "${OUT}/json" "${OUT}/logs" "${OUT}/summary"
exec > >(tee -a "${OUT}/launcher.log") 2>&1

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
    --include-scenes "${scene}" --max-frames-per-scene 49 \
    --lowlight-exposure "${EXPOSURE}" \
    --export-trajectories --num-workers 0 --device cuda --amp --amp-dtype bf16 \
    --output "${output}" \
    > "${OUT}/logs/${scene}.log" 2>&1
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

"${PY}" utils/rank_heldout_mvsec_trajectory_candidates.py \
  --input-dir "${OUT}/json" --output-dir "${OUT}/summary" --dataset-name DL3DV || status=1
if [[ "${status}" -ne 0 ]]; then
  echo "One or more DL3DV screening evaluations failed. Inspect ${OUT}/logs." >&2
  exit "${status}"
fi
touch "${OUT}/COMPLETE"
date -Is
