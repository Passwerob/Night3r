#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 3 ]]; then
  echo "usage: $0 GPU WAIT_SESSION ROLE" >&2
  exit 2
fi

GPU="$1"
WAIT_SESSION="$2"
ROLE="$3"
MAX_USED_MIB="${MAX_USED_MIB:-22000}"

while tmux has-session -t "${WAIT_SESSION}" 2>/dev/null; do
  sleep 60
done

while true; do
  USED_MIB="$(nvidia-smi -i "${GPU}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
  if [[ "${USED_MIB}" =~ ^[0-9]+$ ]] && (( USED_MIB <= MAX_USED_MIB )); then
    break
  fi
  sleep 60
done

cd /data/fcr/data/Night3R
exec bash scripts/run_exp1_single_gpu_ablation_20260712.sh "${GPU}" "${ROLE}"
