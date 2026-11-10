#!/usr/bin/env bash
set -euo pipefail

cd /data/fcr/data/Night3R

PY="${PY:-/data/fcr/miniconda3/envs/stream3r/bin/python}"
GPU="${GPU:-4}"
SCENE_ROOT="${SCENE_ROOT:-/data/fcr/data/Night3R/test/DSEC/screen-000004-dsec}"
EVAL_ROOT="${EVAL_ROOT:-/data/fcr/data/Night3R/outputs/splits/night3r_dsec_extended_holdout_v2/new_scene_only}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/fcr/data/Night3R/outputs/dsec_zurich11b_final_eval_20260720}"
OUTPUT_JSON="${OUTPUT_JSON:-${OUTPUT_DIR}/night3r_screen004_real_966.json}"
LOG="${OUTPUT_JSON%.json}.stdout.log"

CONFIG="${CONFIG:-/data/fcr/data/Night3R/config/train_night3r_mvsec_only_adapt_1k_from_best.yaml}"
CHECKPOINT="${CHECKPOINT:-/data/fcr/data/Night3R/outputs/night3r_mvsec_only_adapt_1k_from_best/last_event_dark3r.pth}"
MAST3R_CKPT="${MAST3R_CKPT:-/data/fcr/data/Night3R/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth}"

mkdir -p "${EVAL_ROOT}" "${OUTPUT_DIR}"
ln -sfn "${SCENE_ROOT}" "${EVAL_ROOT}/screen-000004-dsec"

test -d "${SCENE_ROOT}"
test -f "${CHECKPOINT}"
test -f "${MAST3R_CKPT}"

echo "[$(date -Is)] Evaluating held-out DSEC zurich_city_11_b on GPU ${GPU}" | tee "${LOG}"
echo "scene=${SCENE_ROOT}" | tee -a "${LOG}"
echo "checkpoint=${CHECKPOINT}" | tee -a "${LOG}"

CUDA_VISIBLE_DEVICES="${GPU}" "${PY}" utils/eval_event_dark3r_variant.py \
  --variant event_dark3r \
  --event-mode real \
  --checkpoint "${CHECKPOINT}" \
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
  | tee "${OUTPUT_DIR}/night3r_screen004_real_966.summary.txt"

echo "[$(date -Is)] Evaluation complete: ${OUTPUT_JSON}" | tee -a "${LOG}"
