#!/usr/bin/env bash
set -euo pipefail

cd /data/fcr/data/Night3R

bash scripts/run_final_dsec_darken_depth_control_20260716.sh 4 dark3r e020 128 &
pid4=$!
bash scripts/run_final_dsec_darken_depth_control_20260716.sh 5 dark3r e010 128 &
pid5=$!
bash scripts/run_final_dsec_darken_depth_control_20260716.sh 6 dark3r e005 128 &
pid6=$!

wait "${pid4}"
wait "${pid5}"
wait "${pid6}"

/data/fcr/miniconda3/envs/stream3r/bin/python \
  utils/summarize_final_dsec_darken_controls.py
touch outputs/final_mvsec1k_dsec_darken_controls_20260716/DARK3R_COMPLETE
