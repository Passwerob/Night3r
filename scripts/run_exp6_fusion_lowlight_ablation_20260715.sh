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
INIT_CKPT="outputs/night3r_brg_dl3dv_stage0_adapter_repair_8gpu_3k/last_event_dark3r.pth"
LORALIB_VENDOR="${LORALIB_VENDOR:-/tmp/loralib_vendor}"

case "${ROLE}" in
  concat_mlp)
    CONFIG="config/exp6_fusion_concat_mlp_20260715.yaml"
    RUN_NAME="exp6_fusion_concat_mlp_1gpu_3k_20260715"
    ;;
  additive)
    CONFIG="config/exp6_fusion_additive_20260715.yaml"
    RUN_NAME="exp6_fusion_additive_1gpu_3k_20260715"
    ;;
  srgb_no_noise)
    CONFIG="config/exp6_lowlight_srgb_no_noise_20260715.yaml"
    RUN_NAME="exp6_lowlight_srgb_no_noise_1gpu_3k_20260715"
    ;;
  read_only)
    CONFIG="config/exp6_lowlight_read_only_20260715.yaml"
    RUN_NAME="exp6_lowlight_read_only_1gpu_3k_20260715"
    ;;
  shot_only)
    CONFIG="config/exp6_lowlight_shot_only_20260715.yaml"
    RUN_NAME="exp6_lowlight_shot_only_1gpu_3k_20260715"
    ;;
  *)
    echo "unknown role: ${ROLE}" >&2
    exit 2
    ;;
esac

SAVE_DIR="outputs/${RUN_NAME}"
EVAL_DIR="${SAVE_DIR}/eval_after_train"
mkdir -p "${LORALIB_VENDOR}" "${SAVE_DIR}" "${EVAL_DIR}"
ln -sfn /data/fcr/miniconda3/lib/python3.8/site-packages/loralib "${LORALIB_VENDOR}/loralib"
export PYTHONPATH="${LORALIB_VENDOR}${PYTHONPATH:+:${PYTHONPATH}}"

CUDA_VISIBLE_DEVICES="${GPU}" "${PY}" train_event_dark3r.py \
  --config "${CONFIG}" \
  --save-dir "${SAVE_DIR}" \
  --init-event-checkpoint "${INIT_CKPT}" \
  --model-name-or-path "${MAST3R_CKPT}" \
  --allow-unsafe-checkpoint-load \
  --device cuda \
  --amp \
  --amp-dtype bf16 \
  --max-steps 3000 \
  --seed 0 \
  --log-every 50 \
  > "${SAVE_DIR}/train.log" 2>&1

for spec in \
  "mvsec_current:/data/fcr/data/train_data/MVSEC_outdoor_night2_test" \
  "dsec_current:/data/fcr/data/train_data/DSEC_test"; do
  NAME="${spec%%:*}"
  ROOT="${spec#*:}"
  for MODE in real zero shuffle; do
    CUDA_VISIBLE_DEVICES="${GPU}" "${PY}" utils/eval_event_dark3r_variant.py \
      --variant event_dark3r \
      --checkpoint "${SAVE_DIR}/last_event_dark3r.pth" \
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
      --output "${EVAL_DIR}/${NAME}_${MODE}_128.json" \
      > "${EVAL_DIR}/${NAME}_${MODE}_128.stdout.log" 2>&1
  done
done

echo "[exp6 done] gpu=${GPU} role=${ROLE}"
