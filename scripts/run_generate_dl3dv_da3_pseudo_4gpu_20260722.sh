#!/usr/bin/env bash
set -euo pipefail

cd /data/fcr/data/Night3R

PY="${PY:-/data/fcr/miniconda3/envs/DA3/bin/python}"
MODEL_DIR="${MODEL_DIR:-/data/fcr/data/Depth-Anything-3/checkpoints/DA3-LARGE-1.1}"
SOURCE_ROOT="${SOURCE_ROOT:-/data/fcr/data/train_data/DL3DV}"
TEST_ROOT="${TEST_ROOT:-/data/fcr/data/Night3R/test/DL3DV}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/fcr/data/train_data/DL3DV_DA3_pseudo_v1}"
GPU_LIST="${GPU_LIST:-4 5 6 7}"
CLIPS_PER_SOURCE="${CLIPS_PER_SOURCE:-2}"
NUM_SHARDS=4

mkdir -p "${OUTPUT_ROOT}/logs"

pids=()
shard=0
for gpu in ${GPU_LIST}; do
  log="${OUTPUT_ROOT}/logs/shard_${shard}_gpu${gpu}.log"
  echo "Launching shard ${shard} on GPU ${gpu}: ${log}"
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    export MPLCONFIGDIR="/tmp/night3r_da3_mpl_gpu${gpu}"
    "${PY}" utils/generate_dl3dv_da3_pseudo_depth.py \
      --source-root "${SOURCE_ROOT}" \
      --test-root "${TEST_ROOT}" \
      --output-root "${OUTPUT_ROOT}" \
      --model-dir "${MODEL_DIR}" \
      --device cuda:0 \
      --clips-per-source "${CLIPS_PER_SOURCE}" \
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
  echo "At least one DA3 generation shard failed." >&2
  exit "${status}"
fi

"${PY}" utils/summarize_dl3dv_da3_pseudo.py --root "${OUTPUT_ROOT}"
echo "DA3 pseudo-depth generation complete: ${OUTPUT_ROOT}"
