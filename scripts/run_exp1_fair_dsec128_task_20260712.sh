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
OUT_ROOT="outputs/exp1_fair_dsec128_20260712"
MAST3R_CKPT="checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
LORALIB_VENDOR="${LORALIB_VENDOR:-/tmp/loralib_vendor}"

case "${ROLE}" in
  full_brg)
    CONFIG="config/exp1_brg_full_repeat_20260711.yaml"
    CHECKPOINT="outputs/exp1_ablation_full_repeat_2gpu_3k_20260711/last_event_dark3r.pth"
    ;;
  no_gate_reg)
    CONFIG="config/exp1_brg_no_gate_reg_20260711.yaml"
    CHECKPOINT="outputs/exp1_ablation_no_gate_reg_2gpu_3k_20260711/last_event_dark3r.pth"
    ;;
  scalar_crossattn)
    CONFIG="config/exp1_scalar_crossattn_20260711.yaml"
    CHECKPOINT="outputs/exp1_ablation_scalar_crossattn_2gpu_3k_20260711/last_event_dark3r.pth"
    ;;
  *)
    echo "unknown role: ${ROLE}" >&2
    exit 2
    ;;
esac

mkdir -p "${LORALIB_VENDOR}" "${OUT_ROOT}/${ROLE}"
ln -sfn /data/fcr/miniconda3/lib/python3.8/site-packages/loralib "${LORALIB_VENDOR}/loralib"
export PYTHONPATH="${LORALIB_VENDOR}${PYTHONPATH:+:${PYTHONPATH}}"

OUT_JSON="${OUT_ROOT}/${ROLE}/dsec_current_real_128.json"
if [[ -f "${OUT_JSON}" ]]; then
  echo "[skip] ${OUT_JSON}"
  exit 0
fi

CUDA_VISIBLE_DEVICES="${GPU}" "${PY}" utils/eval_event_dark3r_variant.py \
  --variant event_dark3r \
  --checkpoint "${CHECKPOINT}" \
  --config "${CONFIG}" \
  --data-root /data/fcr/data/train_data/DSEC_test \
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
  --output "${OUT_JSON}" \
  > "${OUT_JSON%.json}.stdout.log" 2>&1

echo "[done] ${OUT_JSON}"
