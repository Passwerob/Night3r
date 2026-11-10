#!/usr/bin/env bash
set -euo pipefail

cd /data/fcr/data/Night3R

PY="${PY:-/data/fcr/miniconda3/envs/stream3r/bin/python}"
OUT="${OUT:-outputs/mvsec_curved_trajectory_compare_sweep_20260726}"
SOURCE_ROOT="${SOURCE_ROOT:-/data/fcr/data/train_data/MVSEC}"
CONFIG="${CONFIG:-config/train_night3r_final_step100_pose_adapt100.yaml}"
GEOMETRY_CKPT="${GEOMETRY_CKPT:-outputs/night3r_final_step100_20260723/event_dark3r_night3r_final.pth}"
POSE_HEAD_CKPT="${POSE_HEAD_CKPT:-outputs/night3r_final_step100_pose_adapt100_20260726/pose_heads/pose_head_step00000100.pth}"
MAST3R_CKPT="${MAST3R_CKPT:-checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth}"
NIGHT_GPU="${NIGHT_GPU:-4}"
MAST_GPU="${MAST_GPU:-5}"
MAX_FRAMES="${MAX_FRAMES:-147}"
CANDIDATES="${CANDIDATES:-c48:163,c70:185,c76:191,c84:199,c92:207,c106:221}"

mkdir -p "${OUT}/inputs" "${OUT}/json" "${OUT}/logs" "${OUT}/summary"

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
  local offset frame
  for offset in 0 1 2; do
    printf -v frame '%06d' "$((start + offset))"
    scenes+=("screen-${frame}-mvsec")
  done
  "${PY}" utils/assemble_mvsec_long_sequence.py \
    --source-root "${SOURCE_ROOT}" \
    --scenes "${scenes[@]}" \
    --output-root "${OUT}/inputs/${name}" \
    --scene-name "${scene}"
}

evaluate_night3r() {
  local name="$1"
  local scene="screen-long-${name}-mvsec"
  local output="${OUT}/json/${name}_night3r.json"
  [[ -s "${output}" ]] && return
  CUDA_VISIBLE_DEVICES="${NIGHT_GPU}" "${PY}" utils/eval_night3r_trajectory.py \
    --variant event_dark3r \
    --checkpoint "${GEOMETRY_CKPT}" \
    --pose-head-checkpoint "${POSE_HEAD_CKPT}" \
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

evaluate_mast3r() {
  local name="$1"
  local scene="screen-long-${name}-mvsec"
  local output="${OUT}/json/${name}_mast3r.json"
  [[ -s "${output}" ]] && return
  CUDA_VISIBLE_DEVICES="${MAST_GPU}" "${PY}" utils/eval_night3r_trajectory.py \
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

for index in "${!names[@]}"; do
  assemble_candidate "${names[index]}" "${starts[index]}"
done

for name in "${names[@]}"; do
  echo "== ${name}: Night3R on GPU ${NIGHT_GPU}, MASt3R on GPU ${MAST_GPU} =="
  evaluate_night3r "${name}" & night_pid=$!
  evaluate_mast3r "${name}" & mast_pid=$!
  status=0
  wait "${night_pid}" || status=1
  wait "${mast_pid}" || status=1
  if [[ "${status}" != "0" ]]; then
    echo "Candidate ${name} has a failed evaluation; continuing the sweep." >&2
  fi
done

night_jsons=()
mast_jsons=()
for name in "${names[@]}"; do
  night_jsons+=("${OUT}/json/${name}_night3r.json")
  mast_jsons+=("${OUT}/json/${name}_mast3r.json")
done
"${PY}" utils/summarize_mvsec_curved_pose_compare.py \
  --names "${names[@]}" \
  --night3r-jsons "${night_jsons[@]}" \
  --mast3r-jsons "${mast_jsons[@]}" \
  --output-dir "${OUT}/summary"

touch "${OUT}/COMPLETE"
date -Is
