#!/usr/bin/env bash
set -euo pipefail

cd /data/fcr/data/Night3R

PIPELINE_ROOT="outputs/night3r_da3_global_pipeline_20260723"
mkdir -p "${PIPELINE_ROOT}"
exec > >(tee -a "${PIPELINE_ROOT}/pipeline.log") 2>&1

variants=(trunk_shape trunk_metric full_shape full_metric)
gpus=(4 5 6 7)
pids=()
for index in "${!variants[@]}"; do
  VARIANT="${variants[$index]}" GPU="${gpus[$index]}" \
    bash scripts/run_night3r_da3_global_variant_1gpu_20260723.sh &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=1
done
if [[ "${status}" -ne 0 ]]; then
  echo "At least one global training job failed." >&2
  exit "${status}"
fi

GPU_LIST="4 5 6 7" \
  bash scripts/run_eval_night3r_da3_global_sweep_4gpu_20260723.sh

touch "${PIPELINE_ROOT}/PIPELINE_COMPLETE"
echo "Global-weight pipeline complete."
