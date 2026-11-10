#!/usr/bin/env bash
set -euo pipefail

cd /data/fcr/data/Night3R

TASK="${1:?usage: $0 mast3r|zero|shuffle}"
PY="${PY:-/data/fcr/miniconda3/envs/stream3r/bin/python}"
GPU="${GPU:?set GPU to an available physical GPU index}"
SCENE_ROOT="${SCENE_ROOT:-/data/fcr/data/Night3R/test/DSEC/screen-000004-dsec}"
EVAL_ROOT="${EVAL_ROOT:-/data/fcr/data/Night3R/outputs/splits/night3r_dsec_extended_holdout_v2/new_scene_only}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/fcr/data/Night3R/outputs/dsec_zurich11b_final_eval_20260720}"

CONFIG="${CONFIG:-/data/fcr/data/Night3R/config/train_night3r_mvsec_only_adapt_1k_from_best.yaml}"
CHECKPOINT="${CHECKPOINT:-/data/fcr/data/Night3R/outputs/night3r_mvsec_only_adapt_1k_from_best/last_event_dark3r.pth}"
MAST3R_CKPT="${MAST3R_CKPT:-/data/fcr/data/Night3R/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth}"

case "${TASK}" in
  mast3r)
    VARIANT="mast3r"
    EVENT_MODE="real"
    OUTPUT_JSON="${OUTPUT_DIR}/mast3r_screen004_rgb_966.json"
    MODEL_ARGS=(--input lowlight)
    ;;
  zero|shuffle)
    VARIANT="event_dark3r"
    EVENT_MODE="${TASK}"
    OUTPUT_JSON="${OUTPUT_DIR}/night3r_screen004_${TASK}_966.json"
    MODEL_ARGS=(--checkpoint "${CHECKPOINT}" --event-mode "${EVENT_MODE}")
    ;;
  *)
    echo "Unknown task: ${TASK}" >&2
    exit 2
    ;;
esac

LOG="${OUTPUT_JSON%.json}.stdout.log"
mkdir -p "${EVAL_ROOT}" "${OUTPUT_DIR}"
ln -sfn "${SCENE_ROOT}" "${EVAL_ROOT}/screen-000004-dsec"

test -d "${SCENE_ROOT}"
test -f "${MAST3R_CKPT}"
if [[ "${VARIANT}" == "event_dark3r" ]]; then
  test -f "${CHECKPOINT}"
fi

echo "[$(date -Is)] task=${TASK} scene=zurich_city_11_b gpu=${GPU}" | tee "${LOG}"

CUDA_VISIBLE_DEVICES="${GPU}" "${PY}" utils/eval_event_dark3r_variant.py \
  --variant "${VARIANT}" \
  "${MODEL_ARGS[@]}" \
  --config "${CONFIG}" \
  --data-root "${EVAL_ROOT}" \
  --model-name-or-path "${MAST3R_CKPT}" \
  --allow-unsafe-checkpoint-load \
  --device cuda \
  --amp \
  --amp-dtype bf16 \
  --batch-size 1 \
  --max-batches 966 \
  --pair-sampling fixed \
  --pair-stride 1 \
  --min-pair-distance 1 \
  --max-pair-distance 1 \
  --no-lowlight-aug \
  --output "${OUTPUT_JSON}" \
  2>&1 | tee -a "${LOG}"

"${PY}" -c "import json; d=json.load(open('${OUTPUT_JSON}')); m=d['dense_metrics']; print('batches=', d['num_batches']); print('raw_absrel=', m['depth_raw_absrel']); print('raw_delta1=', m['depth_raw_delta_1.25']); print('median_absrel=', m['depth_median_aligned_absrel']); print('median_delta1=', m['depth_median_aligned_delta_1.25'])" \
  | tee "${OUTPUT_JSON%.json}.summary.txt"

echo "[$(date -Is)] Complete: ${OUTPUT_JSON}" | tee -a "${LOG}"
