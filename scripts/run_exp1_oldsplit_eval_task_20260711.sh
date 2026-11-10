#!/usr/bin/env bash
set -euo pipefail

cd /data/fcr/data/Night3R

if [[ "$#" -ne 3 ]]; then
  echo "usage: $0 GPU MODEL DATASET" >&2
  exit 2
fi

GPU="$1"
MODEL="$2"
DATASET="$3"

PY="${PY:-/data/fcr/miniconda3/envs/stream3r/bin/python}"
OUT_ROOT="${OUT_ROOT:-outputs/exp1_oldsplit_generalization_20260711}"
MAX_BATCHES="${MAX_BATCHES:-128}"
SEED="${SEED:-17}"
MAST3R_CKPT="${MAST3R_CKPT:-checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth}"
DARK3R_LORA="${DARK3R_LORA:-checkpoints/dark3r_pretrained/weights/0049_lora.pth}"
LORALIB_VENDOR="${LORALIB_VENDOR:-/tmp/loralib_vendor}"

mkdir -p "${LORALIB_VENDOR}" "${OUT_ROOT}/${MODEL}"
ln -sfn /data/fcr/miniconda3/lib/python3.8/site-packages/loralib "${LORALIB_VENDOR}/loralib"
export PYTHONPATH="${LORALIB_VENDOR}${PYTHONPATH:+:${PYTHONPATH}}"

MVSEC_OLD="/data/fcr/data/train_data/MVSEC_outdoor_night2_test_prev_8scene_20260701"
DSEC_OLD="/data/fcr/data/Night3R/outputs/eval_roots/dsec_test_prev_2scene_20260701"

case "${DATASET}" in
  mvsec_old)
    DATA_ROOT="${MVSEC_OLD}"
    SCENE_ARGS=(--max-pairs-per-scene 16)
    ;;
  dsec_old)
    DATA_ROOT="${DSEC_OLD}"
    SCENE_ARGS=()
    ;;
  *)
    echo "unknown dataset: ${DATASET}" >&2
    exit 2
    ;;
esac

case "${MODEL}" in
  mast3r)
    VARIANT="mast3r"
    CONFIG="config/train_night3r_mvsec_only_adapt_1k_from_best.yaml"
    MODEL_ARGS=(--input lowlight)
    ;;
  dark3r_lora)
    VARIANT="dark3r_lora"
    CONFIG="config/train_night3r_mvsec_only_adapt_1k_from_best.yaml"
    MODEL_ARGS=(--dark3r-lora-checkpoint "${DARK3R_LORA}" --input lowlight)
    ;;
  vats300)
    VARIANT="event_dark3r"
    CONFIG="config/train_night3r_mvsec_only_adapt_1k_vats_continue.yaml"
    MODEL_ARGS=(--checkpoint outputs/night3r_1k_vats_continue_300_20260703/last_event_dark3r.pth --event-mode real)
    ;;
  night3r_mixed)
    VARIANT="event_dark3r"
    CONFIG="config/train_night3r_brg_dl3dv_recover_from_stage2.yaml"
    MODEL_ARGS=(--checkpoint outputs/night3r_brg_dl3dv_recover_from_stage2_6k/last_event_dark3r.pth --event-mode real)
    ;;
  *)
    echo "unknown model: ${MODEL}" >&2
    exit 2
    ;;
esac

OUT_JSON="${OUT_ROOT}/${MODEL}/${DATASET}_${MAX_BATCHES}.json"
if [[ -f "${OUT_JSON}" ]]; then
  echo "[skip] ${OUT_JSON}"
  exit 0
fi

CUDA_VISIBLE_DEVICES="${GPU}" "${PY}" utils/eval_event_dark3r_variant.py \
  --variant "${VARIANT}" \
  --config "${CONFIG}" \
  --data-root "${DATA_ROOT}" \
  --model-name-or-path "${MAST3R_CKPT}" \
  --allow-unsafe-checkpoint-load \
  --no-lowlight-aug \
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
  "${SCENE_ARGS[@]}" \
  "${MODEL_ARGS[@]}" \
  --output "${OUT_JSON}" \
  > "${OUT_JSON%.json}.stdout.log" 2>&1

echo "[done] ${OUT_JSON}"
