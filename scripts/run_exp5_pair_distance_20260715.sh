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
OUT_ROOT="outputs/exp5_pair_distance_20260715/${DATASET_NAME}"
MAST3R_CKPT="checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
LORALIB_VENDOR="${LORALIB_VENDOR:-/tmp/loralib_vendor}"
mkdir -p "${LORALIB_VENDOR}" "${OUT_ROOT}"
ln -sfn /data/fcr/miniconda3/lib/python3.8/site-packages/loralib "${LORALIB_VENDOR}/loralib"
export PYTHONPATH="${LORALIB_VENDOR}${PYTHONPATH:+:${PYTHONPATH}}"

for DIST in 1 3 5 7; do
  for ROLE in mast3r dark3r_lora night3r_mvsec_1k night3r_mixed; do
    OUT_JSON="${OUT_ROOT}/${ROLE}_stride${DIST}_128.json"
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
    CUDA_VISIBLE_DEVICES="${GPU}" "${PY}" utils/eval_event_dark3r_variant.py \
      --variant "${VARIANT}" \
      --config "${CONFIG}" \
      --data-root "${DATA_ROOT}" \
      --model-name-or-path "${MAST3R_CKPT}" \
      --allow-unsafe-checkpoint-load \
      --no-lowlight-aug \
      --event-mode real \
      --device cuda \
      --amp \
      --amp-dtype bf16 \
      --batch-size 1 \
      --max-batches 128 \
      --pair-sampling fixed \
      --pair-stride "${DIST}" \
      --min-pair-distance "${DIST}" \
      --max-pair-distance "${DIST}" \
      --num-workers 0 \
      --seed 17 \
      "${EXTRA[@]}" \
      --output "${OUT_JSON}" \
      > "${OUT_ROOT}/${ROLE}_stride${DIST}_128.stdout.log" 2>&1
  done
done

echo "[exp5 pair distance done] ${DATASET_NAME}"

