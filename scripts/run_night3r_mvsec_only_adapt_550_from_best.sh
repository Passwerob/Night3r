#!/usr/bin/env bash
set -euo pipefail

# MVSEC-only adaptation from the current best checkpoint.
# This keeps the protocol identical to the 300/400/500/1k runs and only changes
# the step budget, so the step-count tradeoff curve remains comparable.
export RUN_NAME="${RUN_NAME:-night3r_mvsec_only_adapt_550_from_best}"
export MAX_STEPS="${MAX_STEPS:-550}"
export MASTER_PORT="${MASTER_PORT:-29678}"
export GPU_LIST="${GPU_LIST:-1,2,3,4}"
export EVAL_GPU="${EVAL_GPU:-7}"
export CONFIG="${CONFIG:-config/train_night3r_mvsec_only_adapt_1k_from_best.yaml}"

exec bash scripts/run_night3r_mvsec_only_adapt_1k_from_best.sh
