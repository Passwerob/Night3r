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
OUT_ROOT="outputs/exp5_event_corruption_20260715/${DATASET_NAME}"
CONFIG="config/train_night3r_brg_dl3dv_recover_from_stage2.yaml"
CHECKPOINT="outputs/night3r_brg_dl3dv_recover_from_stage2_6k/last_event_dark3r.pth"
MAST3R_CKPT="checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
LORALIB_VENDOR="${LORALIB_VENDOR:-/tmp/loralib_vendor}"
mkdir -p "${LORALIB_VENDOR}" "${OUT_ROOT}"
ln -sfn /data/fcr/miniconda3/lib/python3.8/site-packages/loralib "${LORALIB_VENDOR}/loralib"
export PYTHONPATH="${LORALIB_VENDOR}${PYTHONPATH:+:${PYTHONPATH}}"

for MODE in real dropout_25 dropout_50 dropout_75 hot_pixels; do
  OUT_JSON="${OUT_ROOT}/${MODE}_128.json"
  if [[ -f "${OUT_JSON}" ]]; then
    echo "[skip] ${OUT_JSON}"
    continue
  fi
  CUDA_VISIBLE_DEVICES="${GPU}" "${PY}" utils/eval_event_dark3r_variant.py \
    --variant event_dark3r \
    --checkpoint "${CHECKPOINT}" \
    --config "${CONFIG}" \
    --data-root "${DATA_ROOT}" \
    --model-name-or-path "${MAST3R_CKPT}" \
    --allow-unsafe-checkpoint-load \
    --event-mode "${MODE}" \
    --no-lowlight-aug \
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
    --output "${OUT_JSON}" \
    > "${OUT_ROOT}/${MODE}_128.stdout.log" 2>&1
done

echo "[exp5 event corruption done] ${DATASET_NAME}"
