#!/usr/bin/env bash
set -euo pipefail

cd /data/fcr/data/Night3R

PY="${PY:-/data/fcr/miniconda3/envs/stream3r/bin/python}"
OUT="${OUT:-outputs/mvsec_long_demo_sweep_20260726}"
SOURCE_ROOT="${SOURCE_ROOT:-/data/fcr/data/train_data/MVSEC}"
CONFIG="${CONFIG:-config/train_night3r_final_step100_pose_adapt100.yaml}"
GEOMETRY_CKPT="${GEOMETRY_CKPT:-outputs/night3r_final_step100_20260723/event_dark3r_night3r_final.pth}"
POSE_HEAD_CKPT="${POSE_HEAD_CKPT:-outputs/night3r_final_step100_pose_adapt100_20260726/pose_heads/pose_head_step00000100.pth}"
MAST3R_CKPT="${MAST3R_CKPT:-checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth}"
CANDIDATES="${CANDIDATES:-c20:135,c31:146,c42:157,c52:167,c64:179,c78:193}"

mkdir -p "${OUT}/inputs" "${OUT}/json" "${OUT}/logs"

names=()
starts=()
IFS=',' read -r -a candidate_specs <<< "${CANDIDATES}"
for candidate_spec in "${candidate_specs[@]}"; do
  candidate_name="${candidate_spec%%:*}"
  candidate_start="${candidate_spec##*:}"
  if [[ -z "${candidate_name}" || ! "${candidate_start}" =~ ^[0-9]+$ ]]; then
    echo "Invalid CANDIDATES entry: ${candidate_spec}" >&2
    exit 2
  fi
  names+=("${candidate_name}")
  starts+=("${candidate_start}")
done

assemble_candidate() {
  local name="$1"
  local start="$2"
  local scene="screen-long-${name}-mvsec"
  local target="${OUT}/inputs/${name}/${scene}"
  if [[ -d "${target}" ]]; then
    return
  fi
  local scenes=()
  for offset in 0 1 2 3; do
    printf -v frame '%06d' "$((start + offset))"
    scenes+=("screen-${frame}-mvsec")
  done
  "${PY}" utils/assemble_mvsec_long_sequence.py \
    --source-root "${SOURCE_ROOT}" \
    --scenes "${scenes[@]}" \
    --output-root "${OUT}/inputs/${name}" \
    --scene-name "${scene}"
}

evaluate_candidate() {
  local gpu="$1"
  local name="$2"
  local scene="screen-long-${name}-mvsec"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" utils/eval_night3r_trajectory.py \
    --variant event_dark3r \
    --checkpoint "${GEOMETRY_CKPT}" \
    --pose-head-checkpoint "${POSE_HEAD_CKPT}" \
    --config "${CONFIG}" \
    --data-root "${OUT}/inputs/${name}" \
    --model-name-or-path "${MAST3R_CKPT}" \
    --allow-unsafe-checkpoint-load \
    --event-mode real --pose-recovery direct \
    --include-scenes "${scene}" --max-frames-per-scene 196 \
    --export-trajectories --num-workers 0 --device cuda --amp --amp-dtype bf16 \
    --output "${OUT}/json/${name}.json" \
    > "${OUT}/logs/${name}.log" 2>&1
}

for index in "${!names[@]}"; do
  assemble_candidate "${names[index]}" "${starts[index]}"
done

for offset in 0 4; do
  pids=()
  for local_index in 0 1 2 3; do
    index=$((offset + local_index))
    [[ "${index}" -lt "${#names[@]}" ]] || continue
    evaluate_candidate "$((4 + local_index))" "${names[index]}" &
    pids+=("$!")
  done
  status=0
  for pid in "${pids[@]}"; do
    wait "${pid}" || status=1
  done
  [[ "${status}" == "0" ]] || exit 1
done

inputs=()
for name in "${names[@]}"; do
  inputs+=("${OUT}/json/${name}.json")
done
"${PY}" utils/summarize_long_pose_demo_sweep.py \
  --inputs "${inputs[@]}" --output-dir "${OUT}/summary"

touch "${OUT}/COMPLETE"
date -Is
