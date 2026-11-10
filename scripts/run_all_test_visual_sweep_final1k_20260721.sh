#!/usr/bin/env bash
set -uo pipefail

ROOT=/data/fcr/data/Night3R
PYTHON=/data/fcr/miniconda3/envs/stream3r/bin/python
OUT="$ROOT/outputs/all_test_visual_sweep_final1k_20260721"
CHECKPOINT="$ROOT/outputs/night3r_mvsec_only_adapt_1k_from_best/last_event_dark3r.pth"
CONFIG="$ROOT/config/train_night3r_mvsec_only_adapt_1k_from_best.yaml"
BACKBONE="$ROOT/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
SWEEP="$ROOT/utils/sweep_all_test_visual_candidates.py"
MERGE="$ROOT/utils/merge_all_test_visual_sweeps.py"

export PYTHONPATH="/tmp/loralib_vendor:$ROOT:${PYTHONPATH:-}"
export MPLCONFIGDIR=/tmp/matplotlib
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
  --allow-unsafe-checkpoint-load
  --resume
)

run_gpu4() {
  CUDA_VISIBLE_DEVICES=4 "$PYTHON" "$SWEEP" "${COMMON[@]}" \
    --dataset-name DSEC --data-root "$ROOT/test/DSEC" --exposure 0.10 \
    --scenes screen-000002-dsec screen-000003-dsec \
    --outdir "$OUT/shards/gpu4_dsec_002_003" \
    >"$OUT/logs/gpu4.log" 2>&1
}

run_gpu5() {
  CUDA_VISIBLE_DEVICES=5 "$PYTHON" "$SWEEP" "${COMMON[@]}" \
    --dataset-name DSEC --data-root "$ROOT/test/DSEC" --exposure 0.10 \
    --scenes screen-000004-dsec \
    --outdir "$OUT/shards/gpu5_dsec_004" \
    >"$OUT/logs/gpu5.log" 2>&1 || return
  CUDA_VISIBLE_DEVICES=5 "$PYTHON" "$SWEEP" "${COMMON[@]}" \
    --dataset-name MVSEC --data-root "$ROOT/test/MVSEC" \
    --scenes screen-000020-mvsec screen-000050-mvsec screen-000095-mvsec \
      screen-000110-mvsec screen-000115-mvsec screen-000122-mvsec \
      screen-000123-mvsec screen-000129-mvsec screen-000144-mvsec \
    --outdir "$OUT/shards/gpu5_mvsec_first9" \
    >>"$OUT/logs/gpu5.log" 2>&1
}

run_gpu6() {
  CUDA_VISIBLE_DEVICES=6 "$PYTHON" "$SWEEP" "${COMMON[@]}" \
    --dataset-name MVSEC --data-root "$ROOT/test/MVSEC" \
    --scenes screen-000156-mvsec screen-000157-mvsec screen-000170-mvsec \
      screen-000200-mvsec screen-000205-mvsec screen-000207-mvsec \
      screen-000208-mvsec screen-000220-mvsec screen-000226-mvsec \
    --outdir "$OUT/shards/gpu6_mvsec_last9" \
    >"$OUT/logs/gpu6.log" 2>&1 || return
  CUDA_VISIBLE_DEVICES=6 "$PYTHON" "$SWEEP" "${COMMON[@]}" \
    --dataset-name DL3DV --data-root "$ROOT/test/DL3DV" --exposure 0.10 \
    --scenes screen-000083 screen-000084 screen-000085 screen-000086 screen-000087 \
      screen-000088 screen-000089 screen-000090 screen-000091 screen-000092 \
      screen-000093 screen-000094 screen-000095 screen-000096 screen-000097 \
      screen-000098 screen-000099 screen-000101 screen-000108 screen-000110 \
    --outdir "$OUT/shards/gpu6_dl3dv_first20" \
    >>"$OUT/logs/gpu6.log" 2>&1
}

run_gpu7() {
  CUDA_VISIBLE_DEVICES=7 "$PYTHON" "$SWEEP" "${COMMON[@]}" \
    --dataset-name DL3DV --data-root "$ROOT/test/DL3DV" --exposure 0.10 \
    --scenes screen-000111 screen-000112 screen-000113 screen-000114 screen-000115 \
      screen-000116 screen-000117 screen-000118 screen-000119 screen-000120 \
      screen-000121 screen-000122 screen-000123 screen-000124 screen-000125 \
      screen-000126 screen-000127 screen-000128 screen-000129 screen-000130 \
    --outdir "$OUT/shards/gpu7_dl3dv_last20" \
    >"$OUT/logs/gpu7.log" 2>&1
}

run_gpu4 & p4=$!
run_gpu5 & p5=$!
run_gpu6 & p6=$!
run_gpu7 & p7=$!

status=0
wait "$p4" || status=1
wait "$p5" || status=1
wait "$p6" || status=1
wait "$p7" || status=1

"$PYTHON" "$MERGE" --root "$OUT" >"$OUT/logs/merge.log" 2>&1 || status=1
exit "$status"
