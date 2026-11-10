#!/usr/bin/env bash
set -euo pipefail

cd /data/fcr/data/Night3R

OUT_ROOT="${OUT_ROOT:-outputs/exp1_follow_20260711_128}"
mkdir -p "${OUT_ROOT}"

PID_FILE="${OUT_ROOT}/run.pid"
LOG_FILE="${OUT_ROOT}/run.log"

if [[ -f "${PID_FILE}" ]]; then
  old_pid="$(cat "${PID_FILE}")"
  if [[ -n "${old_pid}" ]] && kill -0 "${old_pid}" 2>/dev/null; then
    echo "already_running ${old_pid}"
    echo "log ${LOG_FILE}"
    exit 0
  fi
fi

OUT_ROOT="${OUT_ROOT}" MAX_BATCHES=128 \
  nohup bash scripts/run_exp1_follow_20260710.sh > "${LOG_FILE}" 2>&1 &
pid="$!"
echo "${pid}" > "${PID_FILE}"
echo "started ${pid}"
echo "log ${LOG_FILE}"
echo "out ${OUT_ROOT}"
