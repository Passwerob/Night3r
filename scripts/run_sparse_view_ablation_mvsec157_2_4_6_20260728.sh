#!/usr/bin/env bash
set -euo pipefail

# End-to-end sparse-view reconstruction ablation on one event-rich, held-out
# MVSEC scene. All variants cover frames 36--46; only view density changes.

cd /data/fcr/data/Night3R

PY="${PY:-/data/fcr/miniconda3/envs/stream3r/bin/python}"
OUT="${OUT:-outputs/sparse_view_ablation_mvsec157_2_4_6_20260728}"
GPU_LIST="${GPU_LIST:-0,1,2}"
CHECKPOINT="${CHECKPOINT:-outputs/night3r_final_complete_pose_step100_20260726/event_dark3r_night3r_final_with_pose_head.pth}"
POSE_HEAD="${POSE_HEAD:-outputs/night3r_final_step100_pose_adapt100_20260726/pose_heads/pose_head_step00000100.pth}"
CONFIG="${CONFIG:-config/train_night3r_final_step100_pose_adapt100.yaml}"
MAST3R="${MAST3R:-checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth}"
DATA_ROOT="${DATA_ROOT:-/data/fcr/data/Night3R/test/MVSEC}"
SCENE="${SCENE:-screen-000157-mvsec}"
FRAME_SET_2="${FRAME_SET_2:-36,46}"
FRAME_SET_4="${FRAME_SET_4:-36,39,42,46}"
FRAME_SET_6="${FRAME_SET_6:-36,38,40,42,44,46}"
ALIGNMENT="${ALIGNMENT:-pose_head}"
GT_POSE_SCALE_MODE="${GT_POSE_SCALE_MODE:-shared_frame}"

IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
if [[ "${#GPUS[@]}" -lt 1 ]]; then
  echo "GPU_LIST must contain at least one GPU id, got: ${GPU_LIST}" >&2
  exit 2
fi

test -f "${CHECKPOINT}"
test -f "${POSE_HEAD}"
test -f "${MAST3R}"
test -d "${DATA_ROOT}/${SCENE}"

mkdir -p "${OUT}/logs"
exec > >(tee -a "${OUT}/launcher.log") 2>&1

NAMES=(views2 views4 views6)
FRAME_SETS=(
  "${FRAME_SET_2}"
  "${FRAME_SET_4}"
  "${FRAME_SET_6}"
)

run_one() {
  local name="$1" frames="$2" gpu="$3"
  local outdir="${OUT}/${name}"
  if [[ -f "${outdir}/manifest.json" ]]; then
    echo "skip_complete name=${name}"
    return
  fi
  echo "launch name=${name} gpu=${gpu} frames=${frames}"
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 MPLCONFIGDIR=/tmp/matplotlib \
    "${PY}" utils/export_night3r_long_sequence.py \
      --checkpoint "${CHECKPOINT}" \
      --pose-head-checkpoint "${POSE_HEAD}" \
      --config "${CONFIG}" \
      --model-name-or-path "${MAST3R}" \
      --data-root "${DATA_ROOT}" \
      --scene "${SCENE}" \
      --frame-indices "${frames}" \
      --image-size 512 \
      --exposure 0.10 \
      --read-noise-std 0.01 \
      --shot-noise-std 0.01 \
      --lowlight-seed 20260728 \
      --alignment "${ALIGNMENT}" \
      --gt-pose-scale-mode "${GT_POSE_SCALE_MODE}" \
      --conf-quantile 0.55 \
      --min-depth 0.25 \
      --max-depth 80.0 \
      --max-points-per-frame 18000 \
      --voxel-size 0.08 \
      --amp --amp-dtype bf16 --device cuda \
      --allow-unsafe-checkpoint-load \
      --outdir "${outdir}" \
      > "${OUT}/logs/${name}.log" 2>&1
}

pids=()
status=0
for index in "${!NAMES[@]}"; do
  gpu="${GPUS[$((index % ${#GPUS[@]}))]}"
  run_one "${NAMES[index]}" "${FRAME_SETS[index]}" "${gpu}" &
  pids+=("$!")
  if [[ "${#pids[@]}" -ge "${#GPUS[@]}" ]]; then
    for pid in "${pids[@]}"; do
      wait "${pid}" || status=1
    done
    pids=()
  fi
done

for pid in "${pids[@]}"; do
  wait "${pid}" || status=1
done

if [[ "${status}" -ne 0 ]]; then
  echo "sparse_view_ablation_failed" >&2
  exit "${status}"
fi

touch "${OUT}/COMPLETE"
date -Is
