#!/usr/bin/env bash
set -euo pipefail

cd /data/fcr/data/Night3R

OUT_ROOT="outputs/night3r_da3_depth_repair_pipeline_20260723"
mkdir -p "${OUT_ROOT}"
exec > >(tee -a "${OUT_ROOT}/pipeline.log") 2>&1

echo "== DA3 depth-repair pipeline =="
date -Is

VARIANT=shape GPU_LIST=4,5 NPROC=2 MASTER_PORT=29781 \
  bash scripts/run_night3r_da3_depth_repair_2gpu_20260723.sh &
shape_pid=$!

VARIANT=metric GPU_LIST=6,7 NPROC=2 MASTER_PORT=29782 \
  bash scripts/run_night3r_da3_depth_repair_2gpu_20260723.sh &
metric_pid=$!

status=0
wait "${shape_pid}" || status=1
wait "${metric_pid}" || status=1
if [[ "${status}" -ne 0 ]]; then
  echo "At least one DA3 depth-repair training job failed." >&2
  exit "${status}"
fi

GPU_LIST="4 5 6 7" \
  bash scripts/run_eval_night3r_da3_depth_repair_sweep_4gpu_20260723.sh

touch "${OUT_ROOT}/PIPELINE_COMPLETE"
echo "== DA3 depth-repair pipeline complete =="
date -Is
