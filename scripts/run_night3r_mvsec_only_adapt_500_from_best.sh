#!/usr/bin/env bash
set -euo pipefail

# Mid-point MVSEC-only adaptation from the current best checkpoint.
# This reuses the 1k adaptation config so the adaptation-step curve stays clean.
export RUN_NAME="${RUN_NAME:-night3r_mvsec_only_adapt_500_from_best}"
export MAX_STEPS="${MAX_STEPS:-500}"
export MASTER_PORT="${MASTER_PORT:-29676}"
export GPU_LIST="${GPU_LIST:-1,2,3,4}"
export EVAL_GPU="${EVAL_GPU:-7}"
export CONFIG="${CONFIG:-config/train_night3r_mvsec_only_adapt_1k_from_best.yaml}"

exec bash scripts/run_night3r_mvsec_only_adapt_1k_from_best.sh
