#!/usr/bin/env bash
set -uo pipefail

ROOT=/data/fcr/data/Night3R
PYTHON=/data/fcr/miniconda3/envs/stream3r/bin/python
OUT="$ROOT/outputs/dsec_dl3dv_visual_sweep_darker004_20260722"
CHECKPOINT="$ROOT/outputs/night3r_mvsec_only_adapt_1k_from_best/last_event_dark3r.pth"
CONFIG="$ROOT/config/train_night3r_mvsec_only_adapt_1k_from_best.yaml"
BACKBONE="$ROOT/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
SWEEP="$ROOT/utils/sweep_all_test_visual_candidates.py"
MERGE="$ROOT/utils/merge_all_test_visual_sweeps.py"
BUILD_REVIEW="$ROOT/utils/build_pair_review_assets.py"
SERVE_REVIEW="$ROOT/utils/serve_pair_review.py"

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
  --progress-every 25
  --selection-noise-penalty 18.0
  --selection-brightness-penalty 3.0
  --selection-black-penalty 8.0
  --allow-unsafe-checkpoint-load
  --resume
  --read-noise-std 0.00001
  --shot-noise-std 0.00005
)

DSEC_DARK=(
  --exposure 0.004
  --selection-brightness-target 0.028
  --selection-black-floor 0.003
)

DL3DV_DARK=(
  --exposure 0.004
  --selection-brightness-target 0.042
  --selection-black-floor 0.006
)

run_dsec() {
  local gpu=$1 out_name=$2
  shift 2
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$SWEEP" \
    "${COMMON[@]}" "${DSEC_DARK[@]}" \
    --dataset-name DSEC --data-root "$ROOT/test/DSEC" \
    --scenes "$@" --outdir "$OUT/shards/$out_name"
}

run_dl3dv() {
  local gpu=$1 out_name=$2
  shift 2
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$SWEEP" \
    "${COMMON[@]}" "${DL3DV_DARK[@]}" \
    --dataset-name DL3DV --data-root "$ROOT/test/DL3DV" \
    --scenes "$@" --outdir "$OUT/shards/$out_name"
}

queue_gpu4() {
  run_dsec 4 gpu4_dsec_002 screen-000002-dsec || return
  run_dl3dv 4 gpu4_dl3dv_083_085 screen-000083 screen-000084 screen-000085
}

queue_gpu5() {
  run_dsec 5 gpu5_dsec_004 screen-000004-dsec || return
  run_dl3dv 5 gpu5_dl3dv_086_088 screen-000086 screen-000087 screen-000088
}

queue_gpu6() {
  run_dsec 6 gpu6_dsec_003 screen-000003-dsec || return
  run_dl3dv 6 gpu6_dl3dv_089_099 \
    screen-000089 screen-000090 screen-000091 screen-000092 screen-000093 \
    screen-000094 screen-000095 screen-000096 screen-000097 screen-000098 screen-000099
}

queue_gpu7() {
  run_dl3dv 7 gpu7_dl3dv_101_130 \
    screen-000101 screen-000108 screen-000110 screen-000111 screen-000112 \
    screen-000113 screen-000114 screen-000115 screen-000116 screen-000117 \
    screen-000118 screen-000119 screen-000120 screen-000121 screen-000122 \
    screen-000123 screen-000124 screen-000125 screen-000126 screen-000127 \
    screen-000128 screen-000129 screen-000130
}

queue_gpu4 >"$OUT/logs/gpu4.log" 2>&1 & p4=$!
queue_gpu5 >"$OUT/logs/gpu5.log" 2>&1 & p5=$!
queue_gpu6 >"$OUT/logs/gpu6.log" 2>&1 & p6=$!
queue_gpu7 >"$OUT/logs/gpu7.log" 2>&1 & p7=$!

status=0
for pid in "$p4" "$p5" "$p6" "$p7"; do
  wait "$pid" || status=1
done
if [[ "$status" -ne 0 ]]; then
  echo "At least one sweep queue failed; inspect $OUT/logs/gpu*.log" >&2
  touch "$OUT/FAILED"
  exit "$status"
fi

"$PYTHON" "$MERGE" --root "$OUT" >"$OUT/logs/merge.log" 2>&1 || exit 1
"$PYTHON" "$BUILD_REVIEW" --sweep-root "$OUT" --workers 12 \
  >"$OUT/logs/build_review.log" 2>&1 || exit 1
touch "$OUT/COMPLETE"
exec "$PYTHON" "$SERVE_REVIEW" --root "$OUT" --host 0.0.0.0 --port 17897 \
  >>"$OUT/logs/review_server.log" 2>&1
