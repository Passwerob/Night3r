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
TRAIN_SEED=0

case "${ROLE}" in
  full_seed0)
    CONFIG="config/exp3_full_brg_20260714.yaml"
    RUN_NAME="exp3_full_brg_seed0_1gpu_3k_20260714"
    ;;
  full_seed17)
    CONFIG="config/exp3_full_brg_20260714.yaml"
    RUN_NAME="exp3_full_brg_seed17_1gpu_3k_20260714"
    TRAIN_SEED=17
    ;;
  full_seed31)
    CONFIG="config/exp3_full_brg_20260714.yaml"
    RUN_NAME="exp3_full_brg_seed31_1gpu_3k_20260714"
    TRAIN_SEED=31
    ;;
  no_depth_gt)
    CONFIG="config/exp3_loss_no_depth_gt_20260714.yaml"
    RUN_NAME="exp3_loss_no_depth_gt_1gpu_3k_20260714"
    ;;
  no_event_photo)
    CONFIG="config/exp3_loss_no_event_photo_20260714.yaml"
    RUN_NAME="exp3_loss_no_event_photo_1gpu_3k_20260714"
    ;;
  no_dl3dv)
    CONFIG="config/exp3_no_dl3dv_20260714.yaml"
    RUN_NAME="exp3_no_dl3dv_1gpu_3k_20260714"
    ;;
  static_plain)
    CONFIG="config/exp3_loss_static_plain_20260714.yaml"
    RUN_NAME="exp3_loss_static_plain_1gpu_3k_20260714"
    ;;
  no_noise_aug)
    CONFIG="config/exp3_no_noise_aug_20260714.yaml"
    RUN_NAME="exp3_no_noise_aug_1gpu_3k_20260714"
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
  --seed "${TRAIN_SEED}" \
  --log-every 50 \
  > "${SAVE_DIR}/train.log" 2>&1

for spec in \
  "mvsec_current:/data/fcr/data/train_data/MVSEC_outdoor_night2_test" \
  "dsec_current:/data/fcr/data/train_data/DSEC_test"; do
  NAME="${spec%%:*}"
  ROOT="${spec#*:}"
  CUDA_VISIBLE_DEVICES="${GPU}" "${PY}" utils/eval_event_dark3r_variant.py \
    --variant event_dark3r \
    --checkpoint "${SAVE_DIR}/last_event_dark3r.pth" \
    --config "${CONFIG}" \
    --data-root "${ROOT}" \
    --model-name-or-path "${MAST3R_CKPT}" \
    --allow-unsafe-checkpoint-load \
    --event-mode real \
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
    --output "${EVAL_DIR}/${NAME}_real_128.json" \
    > "${EVAL_DIR}/${NAME}_real_128.stdout.log" 2>&1
done

echo "[exp3 done] gpu=${GPU} role=${ROLE} seed=${TRAIN_SEED}"
