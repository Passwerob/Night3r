#!/usr/bin/env bash
set -euo pipefail

cd /data/fcr/data/Night3R

TRAIN_DIR="outputs/night3r_xyz_repair_joint_800_4gpu_20260723"
SWEEP_DIR="outputs/night3r_xyz_repair_checkpoint_sweep_20260723"
CONFIG="config/train_night3r_xyz_repair_joint_800_20260723.yaml"
MODEL="checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
INIT="outputs/night3r_da3_joint_mvsec_dsec_dl3dv_ft1k_4gpu_20260722/event_dark3r_step00001000.pth"
PY="${PY:-/data/fcr/miniconda3/envs/stream3r/bin/python}"

mkdir -p "${SWEEP_DIR}/logs"

echo "Waiting for ${TRAIN_DIR}/TRAIN_COMPLETE"
while [[ ! -f "${TRAIN_DIR}/TRAIN_COMPLETE" ]]; do
  if ! pgrep -f "train_night3r_xyz_repair_joint_800_20260723.yaml" >/dev/null; then
    echo "Training process disappeared before TRAIN_COMPLETE" >&2
    exit 1
  fi
  sleep 30
done

run_one() {
  local gpu="$1"
  local step="$2"
  local checkpoint="$3"
  local label
  label="$(printf '%04d' "${step}")"
  echo "GPU ${gpu}: evaluating step ${step} (${checkpoint})"
  CUDA_VISIBLE_DEVICES="${gpu}" \
    "${PY}" -u utils/sweep_night3r_xyz_repair_checkpoints.py \
      --checkpoint "${checkpoint}" \
      --step-label "${step}" \
      --config "${CONFIG}" \
      --model-name-or-path "${MODEL}" \
      --output "${SWEEP_DIR}/step${label}.json" \
      --device cuda \
      --max-batches 32 \
      --allow-unsafe-checkpoint-load \
      >"${SWEEP_DIR}/logs/step${label}.log" 2>&1
}

worker() {
  local gpu="$1"
  shift
  while [[ "$#" -ge 2 ]]; do
    run_one "${gpu}" "$1" "$2"
    shift 2
  done
}

worker 4 \
  0 "${INIT}" \
  400 "${TRAIN_DIR}/event_dark3r_step00000400.pth" \
  800 "${TRAIN_DIR}/event_dark3r_step00000800.pth" &
pid4=$!

worker 5 \
  100 "${TRAIN_DIR}/event_dark3r_step00000100.pth" \
  500 "${TRAIN_DIR}/event_dark3r_step00000500.pth" &
pid5=$!

worker 6 \
  200 "${TRAIN_DIR}/event_dark3r_step00000200.pth" \
  600 "${TRAIN_DIR}/event_dark3r_step00000600.pth" &
pid6=$!

worker 7 \
  300 "${TRAIN_DIR}/event_dark3r_step00000300.pth" \
  700 "${TRAIN_DIR}/event_dark3r_step00000700.pth" &
pid7=$!

wait "${pid4}" "${pid5}" "${pid6}" "${pid7}"

"${PY}" -u utils/sweep_night3r_xyz_repair_checkpoints.py \
  --summarize-dir "${SWEEP_DIR}" \
  >"${SWEEP_DIR}/tradeoff_summary.log" 2>&1

touch "${SWEEP_DIR}/SWEEP_COMPLETE"
echo "Sweep complete: ${SWEEP_DIR}/tradeoff_summary.md"
