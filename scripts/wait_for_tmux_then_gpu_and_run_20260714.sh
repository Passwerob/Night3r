#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 3 ]]; then
  echo "usage: $0 PREVIOUS_SESSION GPU COMMAND [ARGS...]" >&2
  exit 2
fi

PREVIOUS_SESSION="$1"
GPU="$2"
shift 2

while tmux has-session -t "${PREVIOUS_SESSION}" 2>/dev/null; do
  sleep 30
done

cd /data/fcr/data/Night3R
exec bash scripts/wait_for_gpu_idle_and_run_20260712.sh "${GPU}" "$@"

