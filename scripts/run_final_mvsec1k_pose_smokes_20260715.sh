#!/usr/bin/env bash
set -euo pipefail

cd /data/fcr/data/Night3R

bash scripts/run_final_mvsec1k_pose_control_20260715.sh 4 mvsec real 1 8 smoke_bidirectional_choice &
pid4=$!
bash scripts/run_final_mvsec1k_pose_control_20260715.sh 5 dl3dv real 1 8 smoke_bidirectional_choice &
pid5=$!
bash scripts/run_mast3r_pose_baseline_20260715.sh 6 mvsec 1 8 smoke_bidirectional_choice &
pid6=$!
bash scripts/run_mast3r_pose_baseline_20260715.sh 7 dl3dv 1 8 smoke_bidirectional_choice &
pid7=$!

wait "${pid4}"
wait "${pid5}"
wait "${pid6}"
wait "${pid7}"
