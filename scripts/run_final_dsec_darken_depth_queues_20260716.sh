#!/usr/bin/env bash
set -euo pipefail

cd /data/fcr/data/Night3R

run_method_queue() {
  local gpu="$1"
  local method="$2"
  local exposure_tag
  for exposure_tag in e020 e010 e005; do
    bash scripts/run_final_dsec_darken_depth_control_20260716.sh \
      "${gpu}" "${method}" "${exposure_tag}" 128
  done
}

run_method_queue 4 mast3r &
pid4=$!
run_method_queue 5 real &
pid5=$!
run_method_queue 6 zero &
pid6=$!
run_method_queue 7 shuffle &
pid7=$!

wait "${pid4}"
wait "${pid5}"
wait "${pid6}"
wait "${pid7}"

/data/fcr/miniconda3/envs/stream3r/bin/python \
  utils/summarize_final_dsec_darken_controls.py
touch outputs/final_mvsec1k_dsec_darken_controls_20260716/QUEUE_COMPLETE
