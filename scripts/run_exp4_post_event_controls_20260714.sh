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
MAST3R_CKPT="checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
LORALIB_VENDOR="${LORALIB_VENDOR:-/tmp/loralib_vendor}"

case "${ROLE}" in
  patch_fusion_only)
    CONFIG="config/exp4_patch_fusion_only_20260714.yaml"
    RUN_NAME="exp4_patch_fusion_only_1gpu_3k_20260714"
    ;;
  no_source_bias_only)
    CONFIG="config/exp4_no_source_bias_only_20260714.yaml"
    RUN_NAME="exp4_no_source_bias_only_1gpu_3k_20260714"
    ;;
  no_source_stat_norm_only)
    CONFIG="config/exp4_no_source_stat_norm_only_20260714.yaml"
    RUN_NAME="exp4_no_source_stat_norm_only_1gpu_3k_20260714"
    ;;
  direct_srgb_lowlight)
    CONFIG="config/exp4_direct_srgb_lowlight_20260714.yaml"
    RUN_NAME="exp4_direct_srgb_lowlight_1gpu_3k_20260714"
    ;;
  noise_only_aug)
    CONFIG="config/exp4_noise_only_aug_20260714.yaml"
    RUN_NAME="exp4_noise_only_aug_1gpu_3k_20260714"
    ;;
  fixed_exposure010)
    CONFIG="config/exp4_fixed_exposure010_20260714.yaml"
    RUN_NAME="exp4_fixed_exposure010_1gpu_3k_20260714"
    ;;
  *)
    echo "unknown role: ${ROLE}" >&2
    exit 2
    ;;
esac

SAVE_DIR="outputs/${RUN_NAME}"
CHECKPOINT="${SAVE_DIR}/last_event_dark3r.pth"
OUT_DIR="${SAVE_DIR}/eval_event_controls"
mkdir -p "${LORALIB_VENDOR}" "${OUT_DIR}"
ln -sfn /data/fcr/miniconda3/lib/python3.8/site-packages/loralib "${LORALIB_VENDOR}/loralib"
export PYTHONPATH="${LORALIB_VENDOR}${PYTHONPATH:+:${PYTHONPATH}}"

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "missing checkpoint: ${CHECKPOINT}" >&2
  exit 1
fi

for spec in \
  "mvsec_current:/data/fcr/data/train_data/MVSEC_outdoor_night2_test" \
  "dsec_current:/data/fcr/data/train_data/DSEC_test"; do
  NAME="${spec%%:*}"
  ROOT="${spec#*:}"
  for MODE in zero shuffle; do
    OUT_JSON="${OUT_DIR}/${NAME}_${MODE}_128.json"
    if [[ -f "${OUT_JSON}" ]]; then
      echo "[skip] ${OUT_JSON}"
      continue
    fi
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
      --output "${OUT_JSON}" \
      > "${OUT_DIR}/${NAME}_${MODE}_128.stdout.log" 2>&1
  done
done

echo "[exp4 controls done] gpu=${GPU} role=${ROLE}"
