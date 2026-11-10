#!/usr/bin/env bash
set -euo pipefail

# Strictly held-out MVSEC trajectory sweep for selecting a paper-quality 3D
# trajectory visualization. Each candidate is built only from test/MVSEC.

cd /data/fcr/data/Night3R

PY="${PY:-/data/fcr/miniconda3/envs/stream3r/bin/python}"
OUT="${OUT:-outputs/heldout_mvsec_trajectory_sweep_20260728}"
SOURCE_ROOT="${SOURCE_ROOT:-/data/fcr/data/Night3R/test/MVSEC}"
CONFIG="${CONFIG:-config/train_night3r_final_step100_pose_adapt100.yaml}"
NIGHT3R_CKPT="${NIGHT3R_CKPT:-outputs/night3r_final_complete_pose_step100_20260726/event_dark3r_night3r_final_with_pose_head.pth}"
MAST3R_CKPT="${MAST3R_CKPT:-checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth}"
DARK3R_LORA="${DARK3R_LORA:-checkpoints/dark3r_pretrained/weights/0049_lora.pth}"
VGGT_CKPT="${VGGT_CKPT:-/data/fcr/ckpt/model.pt}"
MAX_FRAMES="${MAX_FRAMES:-98}"
LORALIB_VENDOR="${LORALIB_VENDOR:-/tmp/loralib_vendor}"

mkdir -p "${OUT}/inputs" "${OUT}/json" "${OUT}/logs" "${OUT}/summary"
mkdir -p "${LORALIB_VENDOR}"
ln -sfn /data/fcr/miniconda3/lib/python3.8/site-packages/loralib "${LORALIB_VENDOR}/loralib"
export PYTHONPATH="${LORALIB_VENDOR}${PYTHONPATH:+:${PYTHONPATH}}"
exec > >(tee -a "${OUT}/launcher.log") 2>&1

declare -A CHUNKS=(
  [c122_123]="screen-000122-mvsec screen-000123-mvsec"
  [c156_157]="screen-000156-mvsec screen-000157-mvsec"
  [c207_208]="screen-000207-mvsec screen-000208-mvsec"
)
CANDIDATES=(c122_123 c156_157 c207_208)

assemble_candidate() {
  local name="$1"
  local scene="screen-long-${name}-mvsec"
  local target="${OUT}/inputs/${name}/${scene}"
  if [[ -d "${target}" ]]; then
    return
  fi
  read -r -a chunks <<< "${CHUNKS[${name}]}"
  "${PY}" utils/assemble_mvsec_long_sequence.py \
    --source-root "${SOURCE_ROOT}" \
    --scenes "${chunks[@]}" \
    --output-root "${OUT}/inputs/${name}" \
    --scene-name "${scene}"
}

run_night3r() {
  local name="$1" gpu="$2"
  local scene="screen-long-${name}-mvsec"
  local output="${OUT}/json/${name}_night3r.json"
  [[ -s "${output}" ]] && return
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 "${PY}" \
    utils/eval_night3r_trajectory.py \
    --variant event_dark3r \
    --checkpoint "${NIGHT3R_CKPT}" \
    --config "${CONFIG}" \
    --data-root "${OUT}/inputs/${name}" \
    --model-name-or-path "${MAST3R_CKPT}" \
    --allow-unsafe-checkpoint-load \
    --event-mode real --pose-recovery direct \
    --include-scenes "${scene}" --max-frames-per-scene "${MAX_FRAMES}" \
    --export-trajectories --num-workers 0 --device cuda --amp --amp-dtype bf16 \
    --output "${output}" \
    > "${OUT}/logs/${name}_night3r.log" 2>&1
}

run_mast3r() {
  local name="$1" gpu="$2"
  local scene="screen-long-${name}-mvsec"
  local output="${OUT}/json/${name}_mast3r.json"
  [[ -s "${output}" ]] && return
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 "${PY}" \
    utils/eval_night3r_trajectory.py \
    --variant mast3r \
    --config "${CONFIG}" \
    --data-root "${OUT}/inputs/${name}" \
    --model-name-or-path "${MAST3R_CKPT}" \
    --allow-unsafe-checkpoint-load \
    --pose-recovery pointmap_sim3 \
    --include-scenes "${scene}" --max-frames-per-scene "${MAX_FRAMES}" \
    --export-trajectories --conf-quantile 0.70 --max-pointmap-residual 0.25 \
    --max-pnp-points 50000 --min-pnp-inliers 512 \
    --num-workers 0 --device cuda --amp --amp-dtype bf16 \
    --output "${output}" \
    > "${OUT}/logs/${name}_mast3r.log" 2>&1
}

run_dark3r() {
  local name="$1" gpu="$2"
  local scene="screen-long-${name}-mvsec"
  local output="${OUT}/json/${name}_dark3r.json"
  [[ -s "${output}" ]] && return
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 "${PY}" \
    utils/eval_night3r_trajectory.py \
    --variant dark3r_lora --dark3r-lora-checkpoint "${DARK3R_LORA}" \
    --config "${CONFIG}" \
    --data-root "${OUT}/inputs/${name}" \
    --model-name-or-path "${MAST3R_CKPT}" \
    --allow-unsafe-checkpoint-load \
    --pose-recovery pointmap_sim3 \
    --include-scenes "${scene}" --max-frames-per-scene "${MAX_FRAMES}" \
    --export-trajectories --conf-quantile 0.70 --max-pointmap-residual 0.25 \
    --max-pnp-points 50000 --min-pnp-inliers 512 \
    --num-workers 0 --device cuda --amp --amp-dtype bf16 \
    --output "${output}" \
    > "${OUT}/logs/${name}_dark3r.log" 2>&1
}

run_vggt() {
  local name="$1" gpu="$2"
  local output="${OUT}/json/${name}_vggt.json"
  [[ -s "${output}" ]] && return
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 "${PY}" \
    utils/eval_vggt_trajectory.py \
    --variant vggt --checkpoint "${VGGT_CKPT}" \
    --base-vggt-checkpoint "${VGGT_CKPT}" \
    --config "${CONFIG}" \
    --data-root "${OUT}/inputs/${name}" \
    --max-scenes 1 --max-frames-per-scene "${MAX_FRAMES}" \
    --export-trajectories --num-workers 0 --device cuda --amp --amp-dtype bf16 \
    --output "${output}" \
    > "${OUT}/logs/${name}_vggt.log" 2>&1
}

run_candidate() {
  local name="$1" base_gpu="$2"
  echo "== ${name}: GPUs ${base_gpu}-$((base_gpu + 3)) =="
  run_night3r "${name}" "${base_gpu}" &
  local pid_night=$!
  run_mast3r "${name}" "$((base_gpu + 1))" &
  local pid_mast=$!
  run_dark3r "${name}" "$((base_gpu + 2))" &
  local pid_dark=$!
  run_vggt "${name}" "$((base_gpu + 3))" &
  local pid_vggt=$!
  local status=0
  wait "${pid_night}" || status=1
  wait "${pid_mast}" || status=1
  wait "${pid_dark}" || status=1
  wait "${pid_vggt}" || status=1
  return "${status}"
}

for name in "${CANDIDATES[@]}"; do
  assemble_candidate "${name}"
done

status=0
run_candidate c122_123 0 & pid_a=$!
run_candidate c156_157 4 & pid_b=$!
wait "${pid_a}" || status=1
wait "${pid_b}" || status=1
run_candidate c207_208 0 || status=1

"${PY}" utils/summarize_heldout_mvsec_trajectory_sweep.py \
  --input-dir "${OUT}/json" \
  --candidate-names "${CANDIDATES[@]}" \
  --output-dir "${OUT}/summary" || status=1

if [[ "${status}" -ne 0 ]]; then
  echo "One or more evaluations failed. Inspect ${OUT}/logs." >&2
  exit "${status}"
fi

touch "${OUT}/COMPLETE"
date -Is
