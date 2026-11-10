#!/usr/bin/env bash
set -euo pipefail

cd /data/fcr/data/Night3R

if [[ "$#" -ne 3 ]]; then
  echo "usage: $0 GPU WAIT_SESSION ROLE" >&2
  exit 2
fi

GPU="$1"
WAIT_SESSION="$2"
ROLE="$3"
TASK="scripts/run_exp1_dark005_eval_task_20260711.sh"
D005_OUT="outputs/exp1_dark005_extreme_20260711"
D010_OUT="outputs/exp1_dark010_extreme_20260712"
D005_CFG="config/eval_night3r_mixed_dark005_20260711.yaml"
D010_CFG="config/eval_night3r_mixed_dark010_20260712.yaml"

while tmux has-session -t "${WAIT_SESSION}" 2>/dev/null; do
  sleep 60
done

run_task() {
  local out_root="$1"
  local config="$2"
  local dark_tag="$3"
  local dataset="$4"
  local event_mode="$5"
  OUT_ROOT="${out_root}" CONFIG="${config}" DARK_TAG="${dark_tag}" \
    bash "${TASK}" "${GPU}" night3r_mixed "${dataset}" "${event_mode}"
}

case "${ROLE}" in
  dsec_dark010_real)
    run_task "${D010_OUT}" "${D010_CFG}" dark010 dsec_current real
    ;;
  dsec_dark010_zero)
    run_task "${D010_OUT}" "${D010_CFG}" dark010 dsec_current zero
    ;;
  shuffle)
    run_task "${D005_OUT}" "${D005_CFG}" dark005 mvsec_current shuffle
    run_task "${D010_OUT}" "${D010_CFG}" dark010 mvsec_current shuffle
    ;;
  polarity_temporal)
    run_task "${D005_OUT}" "${D005_CFG}" dark005 mvsec_current polarity_flip
    run_task "${D005_OUT}" "${D005_CFG}" dark005 mvsec_current temporal_reverse
    run_task "${D010_OUT}" "${D010_CFG}" dark010 mvsec_current polarity_flip
    run_task "${D010_OUT}" "${D010_CFG}" dark010 mvsec_current temporal_reverse
    ;;
  *)
    echo "unknown role: ${ROLE}" >&2
    exit 2
    ;;
esac

echo "[missing dark queue done] gpu=${GPU} role=${ROLE}"
