#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/fcr/data/Night3R
PYTHON=/data/fcr/miniconda3/envs/stream3r/bin/python
OUT="$ROOT/outputs/joint_step1000_actual_test_showcase_20260723"
CHECKPOINT="$ROOT/outputs/night3r_da3_joint_mvsec_dsec_dl3dv_ft1k_4gpu_20260722/event_dark3r_step00001000.pth"
CONFIG="$ROOT/config/eval_night3r_da3_ft1k_sweep_20260722.yaml"
BACKBONE="$ROOT/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
SWEEP="$ROOT/utils/sweep_all_test_visual_candidates.py"
MERGE="$ROOT/utils/merge_all_test_visual_sweeps.py"
BUILD_REVIEW="$ROOT/utils/build_pair_review_assets.py"

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
  --image-size 512 512
  --temporal-buckets 8
  --top-per-scene 4
  --amp-dtype bf16
  --device cuda
  --seed 17
  --progress-every 12
  --selection-noise-penalty 16.0
  --selection-brightness-penalty 2.5
  --selection-black-penalty 7.0
  --read-noise-std 0.00003
  --shot-noise-std 0.0001
  --allow-unsafe-checkpoint-load
  --resume
)

CUDA_VISIBLE_DEVICES=0 "$PYTHON" "$SWEEP" \
  "${COMMON[@]}" \
  --dataset-name MVSEC \
  --data-root "$ROOT/test/MVSEC" \
  --scenes screen-000200-mvsec \
  --pair-subsample-stride 1 \
  --outdir "$OUT/shards/mvsec_200" \
  >"$OUT/logs/mvsec.log" 2>&1 &
pid_mvsec=$!

CUDA_VISIBLE_DEVICES=1 "$PYTHON" "$SWEEP" \
  "${COMMON[@]}" \
  --dataset-name DSEC \
  --data-root "$ROOT/test/DSEC" \
  --scenes screen-000003-dsec \
  --pair-subsample-stride 12 \
  --exposure 0.008 \
  --selection-brightness-target 0.040 \
  --selection-black-floor 0.006 \
  --outdir "$OUT/shards/dsec_003" \
  >"$OUT/logs/dsec.log" 2>&1 &
pid_dsec=$!

CUDA_VISIBLE_DEVICES=2 "$PYTHON" "$SWEEP" \
  "${COMMON[@]}" \
  --dataset-name DL3DV \
  --data-root "$ROOT/test/DL3DV" \
  --scenes screen-000083 \
  --pair-subsample-stride 1 \
  --exposure 0.008 \
  --selection-brightness-target 0.060 \
  --selection-black-floor 0.010 \
  --outdir "$OUT/shards/dl3dv_083" \
  >"$OUT/logs/dl3dv.log" 2>&1 &
pid_dl3dv=$!

status=0
for pid in "$pid_mvsec" "$pid_dsec" "$pid_dl3dv"; do
  wait "$pid" || status=1
done
if [[ "$status" -ne 0 ]]; then
  touch "$OUT/FAILED"
  exit "$status"
fi

"$PYTHON" "$MERGE" --root "$OUT" >"$OUT/logs/merge.log" 2>&1
"$PYTHON" "$BUILD_REVIEW" --sweep-root "$OUT" --workers 12 --force \
  >"$OUT/logs/build_review.log" 2>&1
touch "$OUT/COMPLETE"
