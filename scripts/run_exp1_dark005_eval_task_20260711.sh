#!/usr/bin/env bash
set -euo pipefail

cd /data/fcr/data/Night3R

if [[ "$#" -ne 4 ]]; then
  echo "usage: $0 GPU MODEL DATASET EVENT_MODE" >&2
  exit 2
fi

GPU="$1"
MODEL="$2"
DATASET="$3"
EVENT_MODE="$4"

PY="${PY:-/data/fcr/miniconda3/envs/stream3r/bin/python}"
OUT_ROOT="${OUT_ROOT:-outputs/exp1_dark005_extreme_20260711}"
MAX_BATCHES="${MAX_BATCHES:-128}"
SEED="${SEED:-17}"
MAX_USED_MIB="${MAX_USED_MIB:-22000}"
CONFIG="${CONFIG:-config/eval_night3r_mixed_dark005_20260711.yaml}"
DARK_TAG="${DARK_TAG:-dark005}"
MAST3R_CKPT="checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
DARK3R_LORA="checkpoints/dark3r_pretrained/weights/0049_lora.pth"
LORALIB_VENDOR="${LORALIB_VENDOR:-/tmp/loralib_vendor}"

mkdir -p "${LORALIB_VENDOR}" "${OUT_ROOT}/${MODEL}"
ln -sfn /data/fcr/miniconda3/lib/python3.8/site-packages/loralib "${LORALIB_VENDOR}/loralib"
export PYTHONPATH="${LORALIB_VENDOR}${PYTHONPATH:+:${PYTHONPATH}}"

case "${DATASET}" in
  mvsec_current)
    DATA_ROOT="/data/fcr/data/train_data/MVSEC_outdoor_night2_test"
    ;;
  dsec_current)
    DATA_ROOT="/data/fcr/data/train_data/DSEC_test"
    ;;
  *)
    echo "unknown dataset: ${DATASET}" >&2
    exit 2
    ;;
esac

case "${MODEL}" in
  mast3r)
    VARIANT="mast3r"
    MODEL_ARGS=(--input lowlight)
    MODE_NAME="rgb"
    ;;
  dark3r_lora)
    VARIANT="dark3r_lora"
    MODEL_ARGS=(--dark3r-lora-checkpoint "${DARK3R_LORA}" --input lowlight)
    MODE_NAME="rgb"
    ;;
  night3r_mixed)
    VARIANT="event_dark3r"
    MODEL_ARGS=(
      --checkpoint outputs/night3r_brg_dl3dv_recover_from_stage2_6k/last_event_dark3r.pth
      --event-mode "${EVENT_MODE}"
    )
    MODE_NAME="${EVENT_MODE}"
    ;;
  *)
    echo "unknown model: ${MODEL}" >&2
    exit 2
    ;;
esac

OUT_JSON="${OUT_ROOT}/${MODEL}/${DATASET}_${DARK_TAG}_${MODE_NAME}_${MAX_BATCHES}.json"
if [[ -f "${OUT_JSON}" ]]; then
  echo "[skip] ${OUT_JSON}"
  exit 0
fi

# Leave enough headroom for MASt3R/Night3R weights and 512x512 activations.
# This also protects queued jobs if another user occupies the card later.
while true; do
  USED_MIB="$(nvidia-smi -i "${GPU}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
  if [[ "${USED_MIB}" =~ ^[0-9]+$ ]] && (( USED_MIB <= MAX_USED_MIB )); then
    break
  fi
  sleep 60
done

CUDA_VISIBLE_DEVICES="${GPU}" "${PY}" utils/eval_event_dark3r_variant.py \
  --variant "${VARIANT}" \
  --config "${CONFIG}" \
  --data-root "${DATA_ROOT}" \
  --model-name-or-path "${MAST3R_CKPT}" \
  --allow-unsafe-checkpoint-load \
  --device cuda \
  --amp \
  --amp-dtype bf16 \
  --batch-size 1 \
  --max-batches "${MAX_BATCHES}" \
  --pair-sampling fixed \
  --pair-stride 1 \
  --min-pair-distance 1 \
  --max-pair-distance 1 \
  --num-workers 0 \
  --seed "${SEED}" \
  "${MODEL_ARGS[@]}" \
  --output "${OUT_JSON}" \
  > "${OUT_JSON%.json}.stdout.log" 2>&1

echo "[done] ${OUT_JSON}"
