#!/usr/bin/env bash
set -euo pipefail

cd /data/fcr/data/Night3R
TAG="${TAG:-full_6scene32_bidirectional}"

(
  bash scripts/run_final_mvsec1k_pose_control_20260715.sh 4 mvsec real 6 32 "${TAG}"
  bash scripts/run_final_mvsec1k_pose_control_20260715.sh 4 dl3dv real 6 32 "${TAG}"
) &
pid4=$!

(
  bash scripts/run_final_mvsec1k_pose_control_20260715.sh 5 mvsec zero 6 32 "${TAG}"
  bash scripts/run_final_mvsec1k_pose_control_20260715.sh 5 dl3dv zero 6 32 "${TAG}"
) &
pid5=$!

(
  bash scripts/run_final_mvsec1k_pose_control_20260715.sh 6 mvsec shuffle 6 32 "${TAG}"
  bash scripts/run_final_mvsec1k_pose_control_20260715.sh 6 dl3dv shuffle 6 32 "${TAG}"
) &
pid6=$!

(
  bash scripts/run_mast3r_pose_baseline_20260715.sh 7 mvsec 6 32 "${TAG}"
  bash scripts/run_mast3r_pose_baseline_20260715.sh 7 dl3dv 6 32 "${TAG}"
) &
pid7=$!

wait "${pid4}"
wait "${pid5}"
wait "${pid6}"
wait "${pid7}"
