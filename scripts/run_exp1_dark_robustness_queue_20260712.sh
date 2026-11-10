#!/usr/bin/env bash
set -euo pipefail

cd /data/fcr/data/Night3R

if [[ "$#" -ne 2 ]]; then
  echo "usage: $0 GPU ROLE" >&2
  exit 2
fi

GPU="$1"
ROLE="$2"
TASK="scripts/run_exp1_dark005_eval_task_20260711.sh"
D005_OUT="outputs/exp1_dark005_extreme_20260711"
D010_OUT="outputs/exp1_dark010_extreme_20260712"
D005_CFG="config/eval_night3r_mixed_dark005_20260711.yaml"
D010_CFG="config/eval_night3r_mixed_dark010_20260712.yaml"

run_task() {
  local out_root="$1"
  local config="$2"
  local dark_tag="$3"
  local model="$4"
  local dataset="$5"
  local event_mode="$6"
  OUT_ROOT="${out_root}" CONFIG="${config}" DARK_TAG="${dark_tag}" \
    bash "${TASK}" "${GPU}" "${model}" "${dataset}" "${event_mode}"
}

case "${ROLE}" in
  mast3r)
    run_task "${D005_OUT}" "${D005_CFG}" dark005 mast3r dsec_current real
    run_task "${D010_OUT}" "${D010_CFG}" dark010 mast3r mvsec_current real
    run_task "${D010_OUT}" "${D010_CFG}" dark010 mast3r dsec_current real
    ;;
  dark3r_lora)
    run_task "${D005_OUT}" "${D005_CFG}" dark005 dark3r_lora dsec_current real
    run_task "${D010_OUT}" "${D010_CFG}" dark010 dark3r_lora mvsec_current real
    run_task "${D010_OUT}" "${D010_CFG}" dark010 dark3r_lora dsec_current real
    ;;
  mixed_real)
    run_task "${D005_OUT}" "${D005_CFG}" dark005 night3r_mixed dsec_current real
    run_task "${D010_OUT}" "${D010_CFG}" dark010 night3r_mixed mvsec_current real
    run_task "${D010_OUT}" "${D010_CFG}" dark010 night3r_mixed dsec_current real
    run_task "${D005_OUT}" "${D005_CFG}" dark005 night3r_mixed mvsec_current shuffle
    run_task "${D010_OUT}" "${D010_CFG}" dark010 night3r_mixed mvsec_current shuffle
    ;;
  mixed_controls)
    run_task "${D005_OUT}" "${D005_CFG}" dark005 night3r_mixed dsec_current zero
    run_task "${D010_OUT}" "${D010_CFG}" dark010 night3r_mixed mvsec_current zero
    run_task "${D010_OUT}" "${D010_CFG}" dark010 night3r_mixed dsec_current zero
    run_task "${D005_OUT}" "${D005_CFG}" dark005 night3r_mixed mvsec_current polarity_flip
    run_task "${D005_OUT}" "${D005_CFG}" dark005 night3r_mixed mvsec_current temporal_reverse
    run_task "${D010_OUT}" "${D010_CFG}" dark010 night3r_mixed mvsec_current polarity_flip
    run_task "${D010_OUT}" "${D010_CFG}" dark010 night3r_mixed mvsec_current temporal_reverse
    ;;
  *)
    echo "unknown role: ${ROLE}" >&2
    exit 2
    ;;
esac

echo "[queue done] gpu=${GPU} role=${ROLE}"
