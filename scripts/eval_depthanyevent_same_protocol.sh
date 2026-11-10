#!/usr/bin/env bash
set -euo pipefail

cd /data/fcr/data/Night3R

PY="${PY:-/data/fcr/miniconda3/envs/stream3r/bin/python}"
DEPTHANYEVENT_ROOT="${DEPTHANYEVENT_ROOT:-/data/fcr/data/DepthAnyEvent}"
CHECKPOINT="${CHECKPOINT:?Set CHECKPOINT to an official DepthAnyEvent .pth file}"
MODEL_CONFIG="${MODEL_CONFIG:-}"
CACHE="${CACHE:?Set CACHE to the same .pt batch cache used by the compared methods}"
OUTPUT="${OUTPUT:?Set OUTPUT to the result JSON path}"
DEVICE="${DEVICE:-cuda}"
MAX_BATCHES="${MAX_BATCHES:-}"

args=(
  --checkpoint "${CHECKPOINT}"
  --batch-cache "${CACHE}"
  --output "${OUTPUT}"
  --depthanyevent-root "${DEPTHANYEVENT_ROOT}"
  --device "${DEVICE}"
)
if [[ -n "${MAX_BATCHES}" ]]; then
  args+=(--max-batches "${MAX_BATCHES}")
fi
if [[ -n "${MODEL_CONFIG}" ]]; then
  args+=(--model-config "${MODEL_CONFIG}")
fi
if [[ "${DEVICE}" == "cuda" ]]; then
  args+=(--amp --amp-dtype bf16)
fi

"${PY}" utils/eval_depthanyevent_baseline.py "${args[@]}"
