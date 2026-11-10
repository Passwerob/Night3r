#!/usr/bin/env bash
set -euo pipefail

cd /data/fcr/data/Night3R

BASE_CKPT="outputs/night3r_vbrg_no_drop_from_best3k_8gpu_1500/last_event_dark3r.pth"
RUN_SCRIPT="scripts/run_night3r_event_reliability_dsec_holdout_v1_8gpu.sh"

echo "== VGER long run =="
date -Is
echo "RUN_NAME=night3r_vger_floor010_long_from_vbrg1500_4gpu"
echo "CONFIG=config/train_night3r_vger_floor010_long.yaml"
echo "BASE_CKPT=${BASE_CKPT}"
echo "GPU_LIST=4,5,6,7"
echo "MAX_STEPS=4000"
echo "MASTER_PORT=29661"

RUN_NAME="night3r_vger_floor010_long_from_vbrg1500_4gpu" \
CONFIG="config/train_night3r_vger_floor010_long.yaml" \
INIT_CKPT="${BASE_CKPT}" \
GPU_LIST="4,5,6,7" \
NPROC="4" \
EVAL_GPU="4" \
MAX_STEPS="4000" \
MASTER_PORT="29661" \
bash "${RUN_SCRIPT}"

echo "== VGER long run complete =="
date -Is

