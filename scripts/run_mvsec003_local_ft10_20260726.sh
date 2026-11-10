#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/fcr/data/Night3R
PY=/data/fcr/miniconda3/envs/stream3r/bin/python
RUN_ROOT="$ROOT/outputs/mvsec003_local_ft10_20260726"
TRAIN_ROOT="$RUN_ROOT/train_data"
SCENE_ROOT="$TRAIN_ROOT/screen-000157-mvsec"
ASSET_ROOT="$ROOT/outputs/mvsec_selected_003_005_native_20260726/003_screen-000157-mvsec_000036_000037/native_346x260"
SOURCE_ROOT="$ROOT/outputs/mvsec_collapse_event_rich_selected_20260726/03_screen-000157-mvsec_000036_000037/source"
CONFIG="$ROOT/config/train_mvsec003_local_ft10_20260726.yaml"
INIT_CKPT="$ROOT/outputs/night3r_final_step100_20260723/event_dark3r_night3r_final.pth"
BACKBONE="$ROOT/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"

mkdir -p "$SCENE_ROOT/images" "$SCENE_ROOT/events" "$SCENE_ROOT/depth_gt" "$RUN_ROOT/checkpoints"

cp "$ASSET_ROOT/view1_DarkenRGB_native.png" "$SCENE_ROOT/images/000036.png"
cp "$ASSET_ROOT/view2_DarkenRGB_native.png" "$SCENE_ROOT/images/000037.png"
cp "$ASSET_ROOT/view1_Event_original.pt" "$SCENE_ROOT/events/000036.pt"
cp "$ASSET_ROOT/view2_Event_original.pt" "$SCENE_ROOT/events/000037.pt"
cp "$SOURCE_ROOT/depth_gt/000036.pt" "$SCENE_ROOT/depth_gt/000036.pt"
cp "$SOURCE_ROOT/depth_gt/000037.pt" "$SCENE_ROOT/depth_gt/000037.pt"
cp "$SOURCE_ROOT/cameras.yaml" "$SCENE_ROOT/cameras.yaml"
cp "$SOURCE_ROOT/meta.json" "$SCENE_ROOT/meta.json"

cd "$ROOT"
export PYTHONPATH="/tmp/loralib_vendor:$ROOT:${PYTHONPATH:-}"
export MPLCONFIGDIR=/tmp/matplotlib
export NIGHT3R_CPU_THREADS=2
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

{
  echo "== MVSEC 003 local 10-step event-path fit =="
  date -Is
  echo "GPU=${CUDA_VISIBLE_DEVICES:-4}"
  echo "CONFIG=$CONFIG"
  echo "INIT_CKPT=$INIT_CKPT"
  echo "TRAIN_ROOT=$TRAIN_ROOT"

  "$PY" train_event_dark3r.py \
    --config "$CONFIG" \
    --data-root "$TRAIN_ROOT" \
    --save-dir "$RUN_ROOT/checkpoints" \
    --init-event-checkpoint "$INIT_CKPT" \
    --model-name-or-path "$BACKBONE" \
    --allow-unsafe-checkpoint-load \
    --device cuda \
    --amp \
    --amp-dtype bf16 \
    --gradient-checkpointing \
    --max-steps 10 \
    --log-every 1 \
    --seed 17

  test -s "$RUN_ROOT/checkpoints/event_dark3r_step00000010.pth"
  test -s "$RUN_ROOT/checkpoints/last_event_dark3r.pth"
  date -Is
  touch "$RUN_ROOT/TRAIN_COMPLETE"
} 2>&1 | tee "$RUN_ROOT/train.log"
