#!/usr/bin/env bash
set -euo pipefail

cd /data/fcr/data/Night3R

PIPELINE_ROOT="outputs/night3r_da3_partial_rgb_pipeline_20260723"
mkdir -p "${PIPELINE_ROOT}"
exec > >(tee -a "${PIPELINE_ROOT}/pipeline.log") 2>&1

VARIANT=balanced GPU_LIST=4,5 NPROC=2 MASTER_PORT=29784 \
  bash scripts/run_night3r_da3_partial_rgb_2gpu_20260723.sh &
balanced_pid=$!

VARIANT=emphasis GPU_LIST=6,7 NPROC=2 MASTER_PORT=29785 \
  bash scripts/run_night3r_da3_partial_rgb_2gpu_20260723.sh &
emphasis_pid=$!

status=0
wait "${balanced_pid}" || status=1
wait "${emphasis_pid}" || status=1
if [[ "${status}" -ne 0 ]]; then
  echo "At least one partial-RGB training job failed." >&2
  exit "${status}"
fi

GPU_LIST="4 5 6 7" \
  bash scripts/run_eval_night3r_da3_partial_rgb_sweep_4gpu_20260723.sh

touch "${PIPELINE_ROOT}/PIPELINE_COMPLETE"
echo "Partial-RGB pipeline complete."
