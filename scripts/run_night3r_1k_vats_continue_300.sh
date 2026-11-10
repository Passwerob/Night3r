#!/usr/bin/env bash
set -euo pipefail

# Short continuation from the current 1k best checkpoint with VATS enabled.
# Purpose: test whether visibility-aware temporal-bin support preserves the
# existing MVSEC/DSEC metrics while adding a clean temporal-bins ablation point.
export RUN_NAME="${RUN_NAME:-night3r_1k_vats_continue_300}"
export MAX_STEPS="${MAX_STEPS:-300}"
export MASTER_PORT="${MASTER_PORT:-29679}"
export GPU_LIST="${GPU_LIST:-1,2,3,4}"
export EVAL_GPU="${EVAL_GPU:-7}"
export CONFIG="${CONFIG:-config/train_night3r_mvsec_only_adapt_1k_vats_continue.yaml}"
export INIT_CKPT="${INIT_CKPT:-outputs/night3r_mvsec_only_adapt_1k_from_best/last_event_dark3r.pth}"

exec bash scripts/run_night3r_mvsec_only_adapt_1k_from_best.sh
