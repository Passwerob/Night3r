#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/fcr/data/Night3R
RUN="$ROOT/scripts/run_all_test_visual_sweep_step100_stressA_4gpu_20260725.sh"
PYTHON=/data/fcr/miniconda3/envs/stream3r/bin/python

STRESS_A="$ROOT/outputs/all_test_visual_sweep_step100_stressA_20260725"
STRESS_B="$ROOT/outputs/all_test_visual_sweep_step100_stressB_noise4x_20260725"
STRESS_C="$ROOT/outputs/all_test_visual_sweep_step100_stressC_noise15x_20260725"
SUMMARY="$ROOT/outputs/noise_stress_curve_step100_20260725.md"

OUT="$STRESS_B" \
READ_NOISE_STD=0.0003 \
SHOT_NOISE_STD=0.0008 \
bash "$RUN"

OUT="$STRESS_C" \
READ_NOISE_STD=0.001 \
SHOT_NOISE_STD=0.003 \
bash "$RUN"

"$PYTHON" "$ROOT/utils/summarize_noise_stress_curve.py" \
  --sweep Stress-A 0.00008 "$STRESS_A" \
  --sweep Stress-B 0.0003 "$STRESS_B" \
  --sweep Stress-C 0.001 "$STRESS_C" \
  --output "$SUMMARY"
