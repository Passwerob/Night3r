#!/usr/bin/env bash
set -uo pipefail

ROOT=/data/fcr/data/Night3R
PYTHON=/data/fcr/miniconda3/envs/stream3r/bin/python
OUT="$ROOT/outputs/all_test_visual_sweep_ultradark_layered_20260722"
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
  --selection-noise-penalty 16.0
  --selection-brightness-penalty 2.5
  --selection-black-penalty 7.0
  --allow-unsafe-checkpoint-load
  --resume
)

LOW_NOISE=(--read-noise-std 0.00003 --shot-noise-std 0.0001)
DSEC_DARK=(--exposure 0.008 --selection-brightness-target 0.040 --selection-black-floor 0.006)
DL3DV_DARK=(--exposure 0.008 --selection-brightness-target 0.060 --selection-black-floor 0.010)
MVSEC_DARK=(--exposure 0.03 --allow-mvsec-synthetic-darkening --selection-brightness-target 0.018 --selection-black-floor 0.004)

run_sweep() {
  local gpu=$1 dataset=$2 out_name=$3
  shift 3
  local -a dark_args
  case "$dataset" in
    DSEC) dark_args=("${DSEC_DARK[@]}") ;;
    MVSEC) dark_args=("${MVSEC_DARK[@]}") ;;
    DL3DV) dark_args=("${DL3DV_DARK[@]}") ;;
    *) echo "Unknown dataset: $dataset" >&2; return 2 ;;
  esac
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$SWEEP" \
    "${COMMON[@]}" "${LOW_NOISE[@]}" "${dark_args[@]}" \
    --dataset-name "$dataset" --data-root "$ROOT/test/$dataset" \
    --scenes "$@" --outdir "$OUT/shards/$out_name"
}

queue_gpu4() {
  run_sweep 4 DSEC gpu0_dsec_002 screen-000002-dsec
}

queue_gpu5() {
  run_sweep 5 DSEC gpu2_dsec_004 screen-000004-dsec
}

queue_gpu6() {
  run_sweep 6 MVSEC gpu3_mvsec_group_a \
    screen-000050-mvsec screen-000095-mvsec screen-000110-mvsec \
    screen-000115-mvsec screen-000122-mvsec screen-000123-mvsec || return
  run_sweep 6 DL3DV gpu3_dl3dv_group_a \
    screen-000083 screen-000084 screen-000085 screen-000086 screen-000087 screen-000088 || return
  run_sweep 6 MVSEC gpu5_mvsec_group_c \
    screen-000205-mvsec screen-000207-mvsec screen-000208-mvsec \
    screen-000220-mvsec screen-000226-mvsec || return
  run_sweep 6 DL3DV gpu5_dl3dv_group_c \
    screen-000095 screen-000096 screen-000097 screen-000098 \
    screen-000099 screen-000101 screen-000108 || return
  run_sweep 6 DL3DV gpu7_dl3dv_group_e \
    screen-000121 screen-000122 screen-000123 screen-000124 screen-000125 \
    screen-000126 screen-000127 screen-000128 screen-000129 screen-000130
}

queue_gpu7() {
  run_sweep 7 DSEC gpu1_dsec_003 screen-000003-dsec || return
  run_sweep 7 MVSEC gpu1_mvsec_020 screen-000020-mvsec || return
  run_sweep 7 MVSEC gpu4_mvsec_group_b \
    screen-000129-mvsec screen-000144-mvsec screen-000156-mvsec \
    screen-000157-mvsec screen-000170-mvsec screen-000200-mvsec || return
  run_sweep 7 DL3DV gpu4_dl3dv_group_b \
    screen-000089 screen-000090 screen-000091 screen-000092 screen-000093 screen-000094 || return
  run_sweep 7 DL3DV gpu6_dl3dv_group_d \
    screen-000110 screen-000111 screen-000112 screen-000113 screen-000114 \
    screen-000115 screen-000116 screen-000117 screen-000118 screen-000119 screen-000120
}

queue_gpu4 >"$OUT/logs/resume_gpu4.log" 2>&1 & p4=$!
queue_gpu5 >"$OUT/logs/resume_gpu5.log" 2>&1 & p5=$!
queue_gpu6 >"$OUT/logs/resume_gpu6.log" 2>&1 & p6=$!
queue_gpu7 >"$OUT/logs/resume_gpu7.log" 2>&1 & p7=$!

status=0
for pid in "$p4" "$p5" "$p6" "$p7"; do
  wait "$pid" || status=1
done
if [[ "$status" -ne 0 ]]; then
  echo "At least one resumed sweep queue failed; inspect $OUT/logs/resume_gpu*.log" >&2
  touch "$OUT/FAILED"
  exit "$status"
fi

"$PYTHON" "$MERGE" --root "$OUT" >"$OUT/logs/merge.log" 2>&1 || exit 1
"$PYTHON" "$BUILD_REVIEW" --sweep-root "$OUT" --workers 12 \
  >"$OUT/logs/build_review.log" 2>&1 || exit 1
touch "$OUT/COMPLETE"
exec "$PYTHON" "$SERVE_REVIEW" --root "$OUT" --host 0.0.0.0 --port 17896 \
  >>"$OUT/logs/review_server.log" 2>&1
