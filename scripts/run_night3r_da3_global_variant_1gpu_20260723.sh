#!/usr/bin/env bash
set -euo pipefail

cd /data/fcr/data/Night3R

PY="${PY:-/data/fcr/miniconda3/envs/stream3r/bin/python}"
VARIANT="${VARIANT:-trunk_shape}"
GPU="${GPU:-4}"
MAX_STEPS="${MAX_STEPS:-600}"
INIT_CKPT="${INIT_CKPT:-outputs/night3r_three_dataset_balanced_final_20260723/event_dark3r_three_dataset_balanced_final.pth}"
MAST3R_CKPT="${MAST3R_CKPT:-checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth}"

case "${VARIANT}" in
  trunk_shape)
    CONFIG="config/train_night3r_da3_global_trunk_shape_600_20260723.yaml"
    ;;
  trunk_metric)
    CONFIG="config/train_night3r_da3_global_trunk_metric_600_20260723.yaml"
    ;;
  full_shape)
    CONFIG="config/train_night3r_da3_global_full_shape_600_20260723.yaml"
    ;;
  full_metric)
    CONFIG="config/train_night3r_da3_global_full_metric_600_20260723.yaml"
    ;;
  *)
    echo "Unknown VARIANT=${VARIANT}" >&2
    exit 2
    ;;
esac

RUN_NAME="night3r_da3_global_${VARIANT}_600_1gpu_20260723"
SAVE_DIR="outputs/${RUN_NAME}"
mkdir -p "${SAVE_DIR}"
exec > >(tee -a "${SAVE_DIR}/train.log") 2>&1

echo "== Night3R DA3 global sweep: ${VARIANT} =="
date -Is
echo "CONFIG=${CONFIG}"
echo "INIT_CKPT=${INIT_CKPT}"
echo "GPU=${GPU}"

CUDA_VISIBLE_DEVICES="${GPU}" \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"${PY}" train_event_dark3r.py \
  --config "${CONFIG}" \
  --save-dir "${SAVE_DIR}" \
  --init-event-checkpoint "${INIT_CKPT}" \
  --model-name-or-path "${MAST3R_CKPT}" \
  --allow-unsafe-checkpoint-load \
  --device cuda \
  --amp \
  --amp-dtype bf16 \
  --gradient-checkpointing \
  --max-steps "${MAX_STEPS}" \
  --log-every 20

for step in 100 200 300 400 500 600; do
  printf -v checkpoint "${SAVE_DIR}/event_dark3r_step%08d.pth" "${step}"
  test -f "${checkpoint}"
done
test -f "${SAVE_DIR}/last_event_dark3r.pth"
touch "${SAVE_DIR}/TRAIN_COMPLETE"
echo "== ${VARIANT} training complete =="
date -Is

