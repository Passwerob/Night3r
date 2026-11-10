#!/usr/bin/env bash
set -euo pipefail

cd /data/fcr/data/Night3R

if [[ "$#" -lt 3 || "$#" -gt 6 ]]; then
  echo "usage: $0 GPU METHOD EXPOSURE [MAX_SCENES] [MAX_FRAMES] [TAG]" >&2
  exit 2
fi

GPU="$1"
METHOD="$2"
EXPOSURE="$3"
MAX_SCENES="${4:-6}"
MAX_FRAMES="${5:-32}"
TAG="${6:-full_6scene32_bidirectional}"
MAX_POINTMAP_RESIDUAL="${MAX_POINTMAP_RESIDUAL:-0.25}"

PY="/data/fcr/miniconda3/envs/stream3r/bin/python"
CONFIG="config/train_night3r_mvsec_only_adapt_1k_from_best.yaml"
CHECKPOINT="outputs/night3r_mvsec_only_adapt_1k_from_best/last_event_dark3r.pth"
MAST3R_CKPT="checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
ROOT="/data/fcr/data/eval_data/DL3DV"
OUTDIR="outputs/final_mvsec1k_pose_darken_controls_20260716"
LORALIB_VENDOR="${LORALIB_VENDOR:-/tmp/loralib_vendor}"

case "${EXPOSURE}" in
  0.2) EXPOSURE_TAG="e020" ;;
  0.1) EXPOSURE_TAG="e010" ;;
  0.05) EXPOSURE_TAG="e005" ;;
  *) EXPOSURE_TAG="e${EXPOSURE//./p}" ;;
esac

mkdir -p "${LORALIB_VENDOR}" "${OUTDIR}"
ln -sfn /data/fcr/miniconda3/lib/python3.8/site-packages/loralib "${LORALIB_VENDOR}/loralib"
export PYTHONPATH="${LORALIB_VENDOR}${PYTHONPATH:+:${PYTHONPATH}}"

COMMON_ARGS=(
  --config "${CONFIG}"
  --data-root "${ROOT}"
  --model-name-or-path "${MAST3R_CKPT}"
  --allow-unsafe-checkpoint-load
  --pose-recovery pointmap_sim3
  --device cuda
  --amp
  --amp-dtype bf16
  --max-scenes "${MAX_SCENES}"
  --max-frames-per-scene "${MAX_FRAMES}"
  --conf-quantile 0.70
  --max-pointmap-residual "${MAX_POINTMAP_RESIDUAL}"
  --max-pnp-points 50000
  --min-pnp-inliers 512
  --num-workers 0
  --seed 17
  --lowlight-exposure "${EXPOSURE}"
  --lowlight-read-noise-std 0.01
  --lowlight-shot-noise-std 0.01
  --lowlight-gamma 2.2
  --lowlight-seed 20260716
)

if [[ "${METHOD}" == "mast3r" ]]; then
  OUTPUT="${OUTDIR}/dl3dv_mast3r_${EXPOSURE_TAG}_${TAG}.json"
  LOG="${OUTDIR}/dl3dv_mast3r_${EXPOSURE_TAG}_${TAG}.log"
  CUDA_VISIBLE_DEVICES="${GPU}" "${PY}" utils/eval_night3r_trajectory.py \
    --variant mast3r \
    "${COMMON_ARGS[@]}" \
    --output "${OUTPUT}" \
    > "${LOG}" 2>&1
elif [[ "${METHOD}" == "real" || "${METHOD}" == "zero" || "${METHOD}" == "shuffle" ]]; then
  OUTPUT="${OUTDIR}/dl3dv_${METHOD}_${EXPOSURE_TAG}_${TAG}.json"
  LOG="${OUTDIR}/dl3dv_${METHOD}_${EXPOSURE_TAG}_${TAG}.log"
  CUDA_VISIBLE_DEVICES="${GPU}" "${PY}" utils/eval_night3r_trajectory.py \
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

echo "[DL3DV darken pose done] gpu=${GPU} method=${METHOD} exposure=${EXPOSURE} output=${OUTPUT}"
