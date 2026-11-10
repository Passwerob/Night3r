#!/usr/bin/env bash
set -euo pipefail

cd /data/fcr/data/Night3R

PY="${PY:-/data/fcr/miniconda3/envs/stream3r/bin/python}"
OUT="outputs/night3r_rgb_restore_scope_sweep_20260723"
CHECKPOINT="outputs/night3r_da3_joint_mvsec_dsec_dl3dv_ft1k_4gpu_20260722/event_dark3r_step00001000.pth"
CONFIG="config/train_night3r_xyz_repair_joint_800_20260723.yaml"
MODEL="checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"

mkdir -p "${OUT}/logs"

run_scope() {
  local gpu="$1"
  local scope="$2"
  CUDA_VISIBLE_DEVICES="${gpu}" \
    "${PY}" -u utils/sweep_night3r_xyz_repair_checkpoints.py \
      --checkpoint "${CHECKPOINT}" \
      --step-label 0 \
      --config "${CONFIG}" \
      --model-name-or-path "${MODEL}" \
      --output "${OUT}/scope_${scope}.json" \
      --restore-scope "${scope}" \
      --device cuda \
      --max-batches 32 \
      --allow-unsafe-checkpoint-load \
      >"${OUT}/logs/scope_${scope}.log" 2>&1
}

run_scope 4 head2 &
pid4=$!
run_scope 5 heads &
pid5=$!
run_scope 6 geometry &
pid6=$!
run_scope 7 all &
pid7=$!

wait "${pid4}" "${pid5}" "${pid6}" "${pid7}"
touch "${OUT}/SWEEP_COMPLETE"
