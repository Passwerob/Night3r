#!/usr/bin/env bash
set -euo pipefail

cd /data/fcr/data/Night3R

if [[ "$#" -ne 2 ]]; then
  echo "usage: $0 GPU DATASET" >&2
  exit 2
fi

GPU="$1"
DATASET="$2"
PY="${PY:-/data/fcr/miniconda3/envs/stream3r/bin/python}"
CONFIG="config/train_night3r_mvsec_only_adapt_1k_from_best.yaml"
CHECKPOINT="outputs/night3r_mvsec_only_adapt_1k_from_best/last_event_dark3r.pth"
MAST3R_CKPT="checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
OUTDIR="outputs/final_mvsec1k_event_controls_20260715"
LORALIB_VENDOR="${LORALIB_VENDOR:-/tmp/loralib_vendor}"

case "${DATASET}" in
  mvsec)
    ROOT="/data/fcr/data/train_data/MVSEC_outdoor_night2_test"
    ;;
  dsec)
    ROOT="/data/fcr/data/train_data/DSEC_test"
    ;;
  dl3dv)
    ROOT="/data/fcr/data/eval_data/DL3DV"
    ;;
  *)
    echo "unknown dataset: ${DATASET}" >&2
    exit 2
    ;;
esac

mkdir -p "${LORALIB_VENDOR}" "${OUTDIR}"
ln -sfn /data/fcr/miniconda3/lib/python3.8/site-packages/loralib "${LORALIB_VENDOR}/loralib"
export PYTHONPATH="${LORALIB_VENDOR}${PYTHONPATH:+:${PYTHONPATH}}"

for MODE in real zero shuffle; do
  CUDA_VISIBLE_DEVICES="${GPU}" "${PY}" utils/eval_event_dark3r_variant.py \
    --variant event_dark3r \
    --checkpoint "${CHECKPOINT}" \
    --config "${CONFIG}" \
    --data-root "${ROOT}" \
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
    --output "${OUTDIR}/${DATASET}_${MODE}_128.json" \
    > "${OUTDIR}/${DATASET}_${MODE}_128.stdout.log" 2>&1
done

echo "[final checkpoint controls done] gpu=${GPU} dataset=${DATASET}"
