#!/usr/bin/env bash
set -euo pipefail

cd /data/fcr/data/Night3R
if [[ "$#" -ne 1 ]]; then
  echo "usage: $0 GPU" >&2
  exit 2
fi

GPU="$1"
PY="${PY:-/data/fcr/miniconda3/envs/stream3r/bin/python}"
OUT_ROOT="outputs/exp5_efficiency_serial_same_gpu_20260715"
MAST3R_CKPT="checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
LORALIB_VENDOR="${LORALIB_VENDOR:-/tmp/loralib_vendor}"
mkdir -p "${LORALIB_VENDOR}" "${OUT_ROOT}"
ln -sfn /data/fcr/miniconda3/lib/python3.8/site-packages/loralib "${LORALIB_VENDOR}/loralib"
export PYTHONPATH="${LORALIB_VENDOR}${PYTHONPATH:+:${PYTHONPATH}}"

for ROLE in mast3r dark3r_lora night3r_mvsec_1k night3r_mixed; do
  OUT_JSON="${OUT_ROOT}/${ROLE}.json"
  if [[ -f "${OUT_JSON}" ]]; then
    echo "[skip] ${OUT_JSON}"
    continue
  fi
  VARIANT="event_dark3r"
  CONFIG="config/train_night3r_mvsec_only_adapt_1k_from_best.yaml"
  EXTRA=()
  case "${ROLE}" in
    mast3r)
      VARIANT="mast3r"
      ;;
    dark3r_lora)
      VARIANT="dark3r_lora"
      EXTRA+=(--dark3r-lora-checkpoint checkpoints/dark3r_pretrained/weights/0049_lora.pth)
      ;;
    night3r_mvsec_1k)
      EXTRA+=(--checkpoint outputs/night3r_mvsec_only_adapt_1k_from_best/last_event_dark3r.pth)
      ;;
    night3r_mixed)
      CONFIG="config/train_night3r_brg_dl3dv_recover_from_stage2.yaml"
      EXTRA+=(--checkpoint outputs/night3r_brg_dl3dv_recover_from_stage2_6k/last_event_dark3r.pth)
      ;;
  esac
  CUDA_VISIBLE_DEVICES="${GPU}" "${PY}" utils/profile_event_dark3r_efficiency.py \
    --variant "${VARIANT}" \
    --config "${CONFIG}" \
    --model-name-or-path "${MAST3R_CKPT}" \
    --data-root /data/fcr/data/train_data/MVSEC_outdoor_night2_test \
    --warmup 20 \
    --iterations 100 \
    --allow-unsafe-checkpoint-load \
    "${EXTRA[@]}" \
    --output "${OUT_JSON}" \
    > "${OUT_ROOT}/${ROLE}.stdout.log" 2>&1
done

echo "[exp5 efficiency serial done] ${OUT_ROOT}"

