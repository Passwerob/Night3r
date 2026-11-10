#!/usr/bin/env bash
set -euo pipefail

cd /data/fcr/data/Night3R

if [[ "$#" -ne 2 ]]; then
  echo "usage: $0 GPU ROLE" >&2
  exit 2
fi

GPU="$1"
ROLE="$2"
PY="${PY:-/data/fcr/miniconda3/envs/stream3r/bin/python}"
OUT_ROOT="outputs/exp1_efficiency_20260712"
MAST3R_CKPT="checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
LORALIB_VENDOR="${LORALIB_VENDOR:-/tmp/loralib_vendor}"
mkdir -p "${LORALIB_VENDOR}" "${OUT_ROOT}"
ln -sfn /data/fcr/miniconda3/lib/python3.8/site-packages/loralib "${LORALIB_VENDOR}/loralib"
export PYTHONPATH="${LORALIB_VENDOR}${PYTHONPATH:+:${PYTHONPATH}}"

VARIANT="event_dark3r"
CONFIG="config/train_night3r_mvsec_only_adapt_1k_from_best.yaml"
ARGS=()
case "${ROLE}" in
  mast3r)
    VARIANT="mast3r"
    ;;
  dark3r_lora)
    VARIANT="dark3r_lora"
    ARGS+=(--dark3r-lora-checkpoint checkpoints/dark3r_pretrained/weights/0049_lora.pth)
    ;;
  night3r_mvsec_1k)
    ARGS+=(--checkpoint outputs/night3r_mvsec_only_adapt_1k_from_best/last_event_dark3r.pth)
    ;;
  night3r_mixed)
    CONFIG="config/train_night3r_brg_dl3dv_recover_from_stage2.yaml"
    ARGS+=(--checkpoint outputs/night3r_brg_dl3dv_recover_from_stage2_6k/last_event_dark3r.pth)
    ;;
  *)
    echo "unknown role: ${ROLE}" >&2
    exit 2
    ;;
esac

OUT_JSON="${OUT_ROOT}/${ROLE}.json"
if [[ -f "${OUT_JSON}" ]]; then
  echo "[skip] ${OUT_JSON}"
  exit 0
fi

CUDA_VISIBLE_DEVICES="${GPU}" "${PY}" utils/profile_event_dark3r_efficiency.py \
  --variant "${VARIANT}" \
  --config "${CONFIG}" \
  --model-name-or-path "${MAST3R_CKPT}" \
  --data-root /data/fcr/data/train_data/MVSEC_outdoor_night2_test \
  --warmup 10 \
  --iterations 50 \
  --allow-unsafe-checkpoint-load \
  "${ARGS[@]}" \
  --output "${OUT_JSON}" \
  > "${OUT_ROOT}/${ROLE}.stdout.log" 2>&1

echo "[done] ${OUT_JSON}"
