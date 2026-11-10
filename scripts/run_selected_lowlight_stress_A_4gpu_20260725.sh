#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/fcr/data/Night3R
PYTHON=/data/fcr/miniconda3/envs/stream3r/bin/python
EXPORTER="$ROOT/utils/export_selected_lowlight_stress_comparison.py"
OUT="$ROOT/outputs/selected_lowlight_stress_A_20260725"

export PYTHONPATH="/tmp/loralib_vendor:$ROOT:${PYTHONPATH:-}"
export MPLCONFIGDIR=/tmp/matplotlib
export OMP_NUM_THREADS=3
export MKL_NUM_THREADS=3
export OPENBLAS_NUM_THREADS=3

mkdir -p "$OUT/logs"

run_variant() {
  local gpu="$1"
  local variant="$2"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$EXPORTER" \
    --variant "$variant" \
    --datasets DSEC DL3DV \
    --device cuda \
    --amp-dtype bf16 \
    --force \
    >"$OUT/logs/${variant}.log" 2>&1
}

run_variant 4 mast3r &
pid_mast3r=$!
run_variant 5 dark3r &
pid_dark3r=$!
run_variant 6 vggt &
pid_vggt=$!
run_variant 7 night3r &
pid_night3r=$!

status=0
for pid in "$pid_mast3r" "$pid_dark3r" "$pid_vggt" "$pid_night3r"; do
  wait "$pid" || status=1
done
if [[ "$status" -ne 0 ]]; then
  touch "$OUT/FAILED"
  echo "One or more stress-test variants failed; inspect $OUT/logs" >&2
  exit "$status"
fi

"$PYTHON" "$EXPORTER" \
  --finalize \
  --datasets DSEC DL3DV \
  >"$OUT/logs/finalize.log" 2>&1

touch "$OUT/COMPLETE"
echo "Selected low-light stress comparison complete: $OUT"
