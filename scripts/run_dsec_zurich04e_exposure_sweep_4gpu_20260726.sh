#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/fcr/data/Night3R
PYTHON=/data/fcr/miniconda3/envs/stream3r/bin/python
OUT="${OUT:-$ROOT/outputs/dsec_zurich04e_exposure_sweep_step100_20260726}"
CHECKPOINT="${CHECKPOINT:-$ROOT/outputs/night3r_final_step100_20260723/event_dark3r_night3r_final.pth}"
CONFIG="$ROOT/config/eval_night3r_da3_ft1k_sweep_20260722.yaml"
BACKBONE="$ROOT/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
SWEEP="$ROOT/utils/sweep_all_test_visual_candidates.py"
MERGE="$ROOT/utils/merge_all_test_visual_sweeps.py"
BUILD_REVIEW="$ROOT/utils/build_pair_review_assets.py"
SUMMARIZE="$ROOT/utils/summarize_test_sweep_scale.py"

export PYTHONPATH="/tmp/loralib_vendor:$ROOT:${PYTHONPATH:-}"
export MPLCONFIGDIR=/tmp/matplotlib
export NIGHT3R_CPU_THREADS=2
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2

mkdir -p "$OUT/shards" "$OUT/logs"

COMMON=(
  --checkpoint "$CHECKPOINT"
  --config "$CONFIG"
  --model-name-or-path "$BACKBONE"
  --dataset-name DSEC
  --data-root "$ROOT/test/DSEC"
  --scenes screen-000005-dsec
  --image-size 512 512
  --darkening-mode linear_exposure
  --temporal-buckets 8
  --top-per-scene 6
  --amp-dtype bf16
  --device cuda
  --seed 17
  --progress-every 10
  --selection-noise-penalty 18.0
  --selection-brightness-penalty 2.5
  --selection-black-penalty 7.0
  --selection-brightness-target 0.035
  --selection-black-floor 0.006
  --read-noise-std 0.00001
  --shot-noise-std 0.00005
  --pair-subsample-stride 4
  --allow-unsafe-checkpoint-load
  --resume
)

run_level() {
  local gpu="$1"
  local label="$2"
  local exposure="$3"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$SWEEP" "${COMMON[@]}" \
    --exposure "$exposure" \
    --outdir "$OUT/shards/gpu${gpu}_${label}" \
    >"$OUT/logs/gpu${gpu}_${label}.log" 2>&1
}

run_level 4 exp012 0.012 & p4=$!
run_level 5 exp008 0.008 & p5=$!
run_level 6 exp006 0.006 & p6=$!
run_level 7 exp004 0.004 & p7=$!

status=0
wait "$p4" || status=1
wait "$p5" || status=1
wait "$p6" || status=1
wait "$p7" || status=1
if [[ "$status" -ne 0 ]]; then
  touch "$OUT/FAILED"
  echo "At least one exposure sweep failed; inspect $OUT/logs" >&2
  exit "$status"
fi

"$PYTHON" "$MERGE" --root "$OUT" >"$OUT/logs/merge.log" 2>&1
"$PYTHON" "$BUILD_REVIEW" --sweep-root "$OUT" --workers 8 --force \
  >"$OUT/logs/build_review.log" 2>&1
"$PYTHON" "$SUMMARIZE" --sweep-root "$OUT" \
  >"$OUT/logs/scale_summary.log" 2>&1
touch "$OUT/COMPLETE"
echo "DSEC Zurich City 04 E exposure sweep complete: $OUT"
