#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/fcr/data/Night3R
PYTHON=/data/fcr/miniconda3/envs/stream3r/bin/python
EXPORTER="$ROOT/utils/export_selected_dark3r_vggt_baselines.py"
SELECTED="$ROOT/outputs/night3r_selected_pairs_step100_native_rgb_viridis_20260725"
LOGDIR="$SELECTED/metadata/baseline_logs"

export PYTHONPATH="/tmp/loralib_vendor:$ROOT:${PYTHONPATH:-}"
export MPLCONFIGDIR=/tmp/matplotlib
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

mkdir -p "$LOGDIR"

EXTRA_ARGS=()
if [[ "${FORCE:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--force)
fi

CUDA_VISIBLE_DEVICES="${DARK3R_GPU:-4}" "$PYTHON" "$EXPORTER" \
  --selected-root "$SELECTED" \
  --variant dark3r \
  --datasets DSEC DL3DV \
  --device cuda \
  --amp-dtype bf16 \
  "${EXTRA_ARGS[@]}" \
  >"$LOGDIR/dark3r.log" 2>&1 &
dark3r_pid=$!

CUDA_VISIBLE_DEVICES="${VGGT_GPU:-5}" "$PYTHON" "$EXPORTER" \
  --selected-root "$SELECTED" \
  --variant vggt \
  --datasets DSEC DL3DV \
  --device cuda \
  --amp-dtype bf16 \
  "${EXTRA_ARGS[@]}" \
  >"$LOGDIR/vggt.log" 2>&1 &
vggt_pid=$!

status=0
wait "$dark3r_pid" || status=1
wait "$vggt_pid" || status=1
if [[ "$status" -ne 0 ]]; then
  touch "$LOGDIR/FAILED"
  echo "Baseline export failed; inspect $LOGDIR" >&2
  exit "$status"
fi

"$PYTHON" "$EXPORTER" \
  --selected-root "$SELECTED" \
  --finalize \
  --datasets DSEC DL3DV \
  >"$LOGDIR/finalize.log" 2>&1

touch "$LOGDIR/COMPLETE"
echo "Dark3R/VGGT selected-pair supplement complete: $SELECTED"
