#!/usr/bin/env bash
set -euo pipefail

cd /data/fcr/data/Night3R
if [[ "$#" -ne 3 ]]; then
  echo "usage: $0 GPU DATASET_NAME DATA_ROOT" >&2
  exit 2
fi

GPU="$1"
DATASET_NAME="$2"
DATA_ROOT="$3"
PY="${PY:-/data/fcr/miniconda3/envs/stream3r/bin/python}"
OUT_ROOT="outputs/exp5_exposure_curve_20260715/${DATASET_NAME}"
MAST3R_CKPT="checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
MIXED_CKPT="outputs/night3r_brg_dl3dv_recover_from_stage2_6k/last_event_dark3r.pth"
LORALIB_VENDOR="${LORALIB_VENDOR:-/tmp/loralib_vendor}"
mkdir -p "${LORALIB_VENDOR}" "${OUT_ROOT}"
ln -sfn /data/fcr/miniconda3/lib/python3.8/site-packages/loralib "${LORALIB_VENDOR}/loralib"
export PYTHONPATH="${LORALIB_VENDOR}${PYTHONPATH:+:${PYTHONPATH}}"

for TAG in 003 005 010 020 050 100; do
  CONFIG="config/eval_exp5_exposure${TAG}_20260715.yaml"
  for ROLE in mast3r mixed_real mixed_zero; do
    OUT_JSON="${OUT_ROOT}/${ROLE}_exposure${TAG}_128.json"
    if [[ -f "${OUT_JSON}" ]]; then
      echo "[skip] ${OUT_JSON}"
      continue
    fi
    VARIANT="event_dark3r"
    EVENT_MODE="real"
    EXTRA=(--checkpoint "${MIXED_CKPT}")
    if [[ "${ROLE}" == "mast3r" ]]; then
      VARIANT="mast3r"
      EXTRA=()
    elif [[ "${ROLE}" == "mixed_zero" ]]; then
      EVENT_MODE="zero"
    fi
    CUDA_VISIBLE_DEVICES="${GPU}" "${PY}" utils/eval_event_dark3r_variant.py \
      --variant "${VARIANT}" \
      --config "${CONFIG}" \
      --data-root "${DATA_ROOT}" \
      --model-name-or-path "${MAST3R_CKPT}" \
      --allow-unsafe-checkpoint-load \
      --input lowlight \
      --event-mode "${EVENT_MODE}" \
      --device cuda \
      --amp \
      --amp-dtype bf16 \
      --batch-size 1 \
      --max-batches 128 \
      --pair-sampling fixed \
      --pair-stride 1 \
      --min-pair-distance 1 \
      --max-pair-distance 1 \
      --num-workers 0 \
      --seed 17 \
      "${EXTRA[@]}" \
      --output "${OUT_JSON}" \
      > "${OUT_ROOT}/${ROLE}_exposure${TAG}_128.stdout.log" 2>&1
  done
done

echo "[exp5 exposure curve done] ${DATASET_NAME}"

