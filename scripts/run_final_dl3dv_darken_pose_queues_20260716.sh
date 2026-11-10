#!/usr/bin/env bash
set -euo pipefail

cd /data/fcr/data/Night3R
TAG="${TAG:-full_6scene32_bidirectional}"

run_method_queue() {
  local gpu="$1"
  local method="$2"
  local exposure
  for exposure in 0.2 0.1 0.05; do
    bash scripts/run_final_dl3dv_darken_pose_control_20260716.sh \
      "${gpu}" "${method}" "${exposure}" 6 32 "${TAG}"
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

TAG="${TAG}" /data/fcr/miniconda3/envs/stream3r/bin/python \
  utils/summarize_final_darken_pose_controls.py
touch "outputs/final_mvsec1k_pose_darken_controls_20260716/QUEUE_COMPLETE_${TAG}"
