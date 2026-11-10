#!/usr/bin/env bash
set -euo pipefail

cd /data/fcr/data/Night3R

BASE_CKPT="outputs/night3r_vbrg_no_drop_from_best3k_8gpu_1500/last_event_dark3r.pth"
RUN_SCRIPT="scripts/run_night3r_event_reliability_dsec_holdout_v1_8gpu.sh"

run_short() {
  local run_name="$1"
  local config="$2"
  local max_steps="$3"
  local port="$4"

  echo "== VGER short run =="
  date -Is
  echo "RUN_NAME=${run_name}"
  echo "CONFIG=${config}"
  echo "BASE_CKPT=${BASE_CKPT}"
  echo "GPU_LIST=0,1,2,3"
  echo "MAX_STEPS=${max_steps}"
  echo "MASTER_PORT=${port}"

  RUN_NAME="${run_name}" \
  CONFIG="${config}" \
  INIT_CKPT="${BASE_CKPT}" \
  GPU_LIST="0,1,2,3" \
  NPROC="4" \
  EVAL_GPU="0" \
  MAX_STEPS="${max_steps}" \
  MASTER_PORT="${port}" \
  bash "${RUN_SCRIPT}"
}

# Short A: conservative visible-region suppression.
run_short \
  "night3r_vger_floor015_short_from_vbrg1500_4gpu" \
  "config/train_night3r_vger_floor015_short.yaml" \
  "1000" \
  "29651"

# Short B: aggressive visible-region suppression, starting from the same base
# checkpoint so the comparison isolates the VGER floor/gamma choice.
run_short \
  "night3r_vger_floor005_short_from_vbrg1500_4gpu" \
  "config/train_night3r_vger_floor005_short.yaml" \
  "1000" \
  "29652"

echo "== VGER short queue complete =="
date -Is

