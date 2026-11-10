#!/usr/bin/env bash
set -euo pipefail

cd /data/fcr/data/Night3R

if [[ "$#" -lt 3 || "$#" -gt 4 ]]; then
  echo "usage: $0 GPU METHOD EXPOSURE_TAG [MAX_BATCHES]" >&2
  exit 2
fi

GPU="$1"
METHOD="$2"
EXPOSURE_TAG="$3"
MAX_BATCHES="${4:-128}"

PY="/data/fcr/miniconda3/envs/stream3r/bin/python"
CHECKPOINT="outputs/night3r_mvsec_only_adapt_1k_from_best/last_event_dark3r.pth"
MAST3R_CKPT="checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
DARK3R_LORA="checkpoints/dark3r_pretrained/weights/0049_lora.pth"
ROOT="/data/fcr/data/train_data/DSEC_test"
OUTDIR="outputs/final_mvsec1k_dsec_darken_controls_20260716"
LORALIB_VENDOR="${LORALIB_VENDOR:-/tmp/loralib_vendor}"

case "${EXPOSURE_TAG}" in
  e020) CONFIG="config/eval_final_mvsec1k_dsec_darken_e020_20260716.yaml" ;;
  e010) CONFIG="config/eval_final_mvsec1k_dsec_darken_e010_20260716.yaml" ;;
  e005) CONFIG="config/eval_final_mvsec1k_dsec_darken_e005_20260716.yaml" ;;
  *) echo "unknown exposure tag: ${EXPOSURE_TAG}" >&2; exit 2 ;;
esac

mkdir -p "${LORALIB_VENDOR}" "${OUTDIR}"
ln -sfn /data/fcr/miniconda3/lib/python3.8/site-packages/loralib "${LORALIB_VENDOR}/loralib"
export PYTHONPATH="${LORALIB_VENDOR}${PYTHONPATH:+:${PYTHONPATH}}"

COMMON_ARGS=(
  --config "${CONFIG}"
  --data-root "${ROOT}"
  --model-name-or-path "${MAST3R_CKPT}"
  --allow-unsafe-checkpoint-load
  --device cuda
  --amp
  --amp-dtype bf16
  --batch-size 1
  --max-batches "${MAX_BATCHES}"
  --pair-sampling fixed
  --pair-stride 1
  --min-pair-distance 1
  --max-pair-distance 1
  --num-workers 0
  --seed 17
)

OUTPUT="${OUTDIR}/dsec_${METHOD}_${EXPOSURE_TAG}_${MAX_BATCHES}.json"
LOG="${OUTPUT%.json}.log"

if [[ "${METHOD}" == "mast3r" ]]; then
  CUDA_VISIBLE_DEVICES="${GPU}" "${PY}" utils/eval_event_dark3r_variant.py \
    --variant mast3r \
    --input lowlight \
    "${COMMON_ARGS[@]}" \
    --output "${OUTPUT}" \
    > "${LOG}" 2>&1
elif [[ "${METHOD}" == "dark3r" ]]; then
  CUDA_VISIBLE_DEVICES="${GPU}" "${PY}" utils/eval_event_dark3r_variant.py \
    --variant dark3r_lora \
    --dark3r-lora-checkpoint "${DARK3R_LORA}" \
    --input lowlight \
    "${COMMON_ARGS[@]}" \
    --output "${OUTPUT}" \
    > "${LOG}" 2>&1
elif [[ "${METHOD}" == "real" || "${METHOD}" == "zero" || "${METHOD}" == "shuffle" ]]; then
  CUDA_VISIBLE_DEVICES="${GPU}" "${PY}" utils/eval_event_dark3r_variant.py \
    --variant event_dark3r \
    --checkpoint "${CHECKPOINT}" \
    --event-mode "${METHOD}" \
    "${COMMON_ARGS[@]}" \
    --output "${OUTPUT}" \
    > "${LOG}" 2>&1
else
  echo "unknown method: ${METHOD}" >&2
  exit 2
fi

echo "[DSEC darken depth done] gpu=${GPU} method=${METHOD} exposure=${EXPOSURE_TAG} output=${OUTPUT}"
