#!/usr/bin/env bash
set -euo pipefail

cd /data/fcr/data/Night3R

RUN_SCRIPT="scripts/run_night3r_event_reliability_dsec_holdout_v1_8gpu.sh"

echo "== VGER from-MASt3R 8-GPU long run =="
date -Is
echo "RUN_NAME=night3r_vger_floor010_from_mast3r_8gpu_long"
echo "CONFIG=config/train_night3r_vger_floor010_long.yaml"
echo "INIT_CKPT=<none; train event branch from MASt3R initialization>"
echo "GPU_LIST=0,1,2,3,4,5,6,7"
echo "MAX_STEPS=8000"
echo "MASTER_PORT=29671"

RUN_NAME="night3r_vger_floor010_from_mast3r_8gpu_long" \
CONFIG="config/train_night3r_vger_floor010_long.yaml" \
INIT_CKPT="" \
GPU_LIST="0,1,2,3,4,5,6,7" \
NPROC="8" \
EVAL_GPU="7" \
MAX_STEPS="8000" \
MASTER_PORT="29671" \
bash "${RUN_SCRIPT}"

echo "== VGER from-MASt3R 8-GPU long run complete =="
date -Is

