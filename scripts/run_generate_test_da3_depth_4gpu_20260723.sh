#!/usr/bin/env bash
set -euo pipefail

cd /data/fcr/data/Night3R

PY="${PY:-/data/fcr/miniconda3/envs/DA3/bin/python}"
MODEL_DIR="${MODEL_DIR:-/data/fcr/data/Depth-Anything-3/checkpoints/DA3-LARGE-1.1}"
TEST_ROOT="${TEST_ROOT:-/data/fcr/data/Night3R/test}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/fcr/data/Night3R/outputs/test_dl3dv_da3_depth_20260723}"
GPU_LIST="${GPU_LIST:-4 5 6 7}"
NUM_SHARDS=4

mkdir -p "${OUTPUT_ROOT}/logs"

pids=()
shard=0
for gpu in ${GPU_LIST}; do
  log="${OUTPUT_ROOT}/logs/shard_${shard}_gpu${gpu}.log"
  echo "Launching DA3 test shard ${shard} on GPU ${gpu}: ${log}"
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    export MPLCONFIGDIR="/tmp/night3r_test_da3_mpl_gpu${gpu}"
    export OMP_NUM_THREADS=2
    export MKL_NUM_THREADS=2
    "${PY}" utils/generate_test_da3_depth.py \
      --test-root "${TEST_ROOT}" \
      --output-root "${OUTPUT_ROOT}" \
      --model-dir "${MODEL_DIR}" \
      --datasets DL3DV \
      --device cuda:0 \
      --window-size 8 \
      --window-overlap 2 \
      --process-res 504 \
      --target-size 512 512 \
      --num-shards "${NUM_SHARDS}" \
      --shard-index "${shard}"
  ) >"${log}" 2>&1 &
  pids+=("$!")
  shard=$((shard + 1))
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

if [[ "${status}" -ne 0 ]]; then
  touch "${OUTPUT_ROOT}/FAILED"
  echo "At least one DA3 test-depth shard failed." >&2
  exit "${status}"
fi

"${PY}" utils/summarize_test_da3_depth.py --root "${OUTPUT_ROOT}" \
  >"${OUTPUT_ROOT}/logs/summary.log" 2>&1
touch "${OUTPUT_ROOT}/COMPLETE"
echo "DA3 test-depth generation complete: ${OUTPUT_ROOT}"
