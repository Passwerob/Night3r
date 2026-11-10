#!/usr/bin/env bash
set -euo pipefail

# Short MVSEC-only adaptation from the current best checkpoint.
# This intentionally reuses the 1k adaptation config so the main variable is
# the number of optimization steps.
export RUN_NAME="${RUN_NAME:-night3r_mvsec_only_adapt_300_from_best}"
export MAX_STEPS="${MAX_STEPS:-300}"
export MASTER_PORT="${MASTER_PORT:-29675}"
export GPU_LIST="${GPU_LIST:-1,2,3,4}"
export EVAL_GPU="${EVAL_GPU:-7}"
export CONFIG="${CONFIG:-config/train_night3r_mvsec_only_adapt_1k_from_best.yaml}"

exec bash scripts/run_night3r_mvsec_only_adapt_1k_from_best.sh
