#!/usr/bin/env bash
set -euo pipefail

cd /data/fcr/data/Night3R

if [[ "$#" -lt 3 || "$#" -gt 6 ]]; then
  echo "usage: $0 GPU DATASET EVENT_MODE [MAX_SCENES] [MAX_FRAMES] [TAG]" >&2
  exit 2
fi

GPU="$1"
DATASET="$2"
EVENT_MODE="$3"
MAX_SCENES="${4:-6}"
MAX_FRAMES="${5:-32}"
TAG="${6:-full}"
MAX_POINTMAP_RESIDUAL="${MAX_POINTMAP_RESIDUAL:-0.25}"

PY="/data/fcr/miniconda3/envs/stream3r/bin/python"
CONFIG="config/train_night3r_mvsec_only_adapt_1k_from_best.yaml"
CHECKPOINT="outputs/night3r_mvsec_only_adapt_1k_from_best/last_event_dark3r.pth"
MAST3R_CKPT="checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
OUTDIR="outputs/final_mvsec1k_pose_controls_20260715"
LORALIB_VENDOR="${LORALIB_VENDOR:-/tmp/loralib_vendor}"

case "${DATASET}" in
  mvsec)
    ROOT="/data/fcr/data/train_data/MVSEC_outdoor_night2_test"
    ;;
  dl3dv)
    ROOT="/data/fcr/data/eval_data/DL3DV"
    ;;
  *)
    echo "pose GT is unavailable or dataset is unknown: ${DATASET}" >&2
    exit 2
    ;;
esac

mkdir -p "${LORALIB_VENDOR}" "${OUTDIR}"
ln -sfn /data/fcr/miniconda3/lib/python3.8/site-packages/loralib "${LORALIB_VENDOR}/loralib"
export PYTHONPATH="${LORALIB_VENDOR}${PYTHONPATH:+:${PYTHONPATH}}"

OUTPUT="${OUTDIR}/${DATASET}_${EVENT_MODE}_${TAG}.json"
LOG="${OUTDIR}/${DATASET}_${EVENT_MODE}_${TAG}.log"

CUDA_VISIBLE_DEVICES="${GPU}" "${PY}" utils/eval_night3r_trajectory.py \
  --variant event_dark3r \
  --checkpoint "${CHECKPOINT}" \
  --config "${CONFIG}" \
  --data-root "${ROOT}" \
  --model-name-or-path "${MAST3R_CKPT}" \
  --allow-unsafe-checkpoint-load \
  --event-mode "${EVENT_MODE}" \
  --pose-recovery pointmap_sim3 \
  --device cuda \
  --amp \
  --amp-dtype bf16 \
  --max-scenes "${MAX_SCENES}" \
  --max-frames-per-scene "${MAX_FRAMES}" \
  --conf-quantile 0.70 \
  --max-pointmap-residual "${MAX_POINTMAP_RESIDUAL}" \
  --max-pnp-points 50000 \
  --min-pnp-inliers 512 \
  --num-workers 0 \
  --seed 17 \
  --output "${OUTPUT}" \
  > "${LOG}" 2>&1

echo "[pose control done] gpu=${GPU} dataset=${DATASET} event=${EVENT_MODE} output=${OUTPUT}"
