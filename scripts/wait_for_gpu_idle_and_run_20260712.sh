#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 2 ]]; then
  echo "usage: $0 GPU COMMAND [ARGS...]" >&2
  exit 2
fi

GPU="$1"
shift
MAX_USED_MIB="${MAX_USED_MIB:-22000}"
MAX_UTIL="${MAX_UTIL:-10}"
STABLE_POLLS="${STABLE_POLLS:-3}"
stable=0

while (( stable < STABLE_POLLS )); do
  read -r USED UTIL < <(
    nvidia-smi -i "${GPU}" \
      --query-gpu=memory.used,utilization.gpu \
      --format=csv,noheader,nounits | tr -d ' ' | tr ',' ' '
  )
  if [[ "${USED}" =~ ^[0-9]+$ ]] && [[ "${UTIL}" =~ ^[0-9]+$ ]] \
      && (( USED <= MAX_USED_MIB )) && (( UTIL <= MAX_UTIL )); then
    stable=$((stable + 1))
  else
    stable=0
  fi
  if (( stable < STABLE_POLLS )); then
    sleep 20
  fi
done

cd /data/fcr/data/Night3R
exec "$@"
