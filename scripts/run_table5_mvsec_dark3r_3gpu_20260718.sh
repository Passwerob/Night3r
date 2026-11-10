#!/usr/bin/env bash
set -euo pipefail

cd /data/fcr/data/Night3R

PY=/data/fcr/miniconda3/envs/stream3r/bin/python
OUTDIR=outputs/darken_cutoff_qualitative_20260716
MAST3R_CKPT=checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth
DARK3R_LORA=checkpoints/dark3r_pretrained/weights/0049_lora.pth
MVSEC_ROOT=/data/fcr/data/train_data/MVSEC_outdoor_night2_test
LORALIB_VENDOR=${LORALIB_VENDOR:-/tmp/loralib_vendor}

mkdir -p "${LORALIB_VENDOR}" "${OUTDIR}"
ln -sfn /data/fcr/miniconda3/lib/python3.8/site-packages/loralib "${LORALIB_VENDOR}/loralib"
export PYTHONPATH="${LORALIB_VENDOR}${PYTHONPATH:+:${PYTHONPATH}}"

run_one() {
  local gpu="$1" exposure="$2"
  local config="config/eval_final_mvsec1k_dsec_darken_${exposure}_20260716.yaml"
  local output="${OUTDIR}/mvsec_dark3r_${exposure}_128.json"
  local log="${output%.json}.log"

  if [[ -s "${output}" ]]; then
    echo "[skip] ${output}"
    return
  fi

  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 "${PY}" \
    utils/eval_event_dark3r_variant.py \
    --variant dark3r_lora \
    --dark3r-lora-checkpoint "${DARK3R_LORA}" \
    --input lowlight \
    --config "${config}" \
    --data-root "${MVSEC_ROOT}" \
    --model-name-or-path "${MAST3R_CKPT}" \
    --allow-unsafe-checkpoint-load \
    --device cuda --amp --amp-dtype bf16 \
    --batch-size 1 --max-batches 128 \
    --pair-sampling fixed --pair-stride 1 \
    --min-pair-distance 1 --max-pair-distance 1 \
    --num-workers 0 --seed 17 \
    --output "${output}" > "${log}" 2>&1

  echo "[done] gpu=${gpu} exposure=${exposure} output=${output}"
}

run_one 4 e020 & pid4=$!
run_one 5 e010 & pid5=$!
run_one 6 e005 & pid6=$!

failed=0
for pid in "${pid4}" "${pid5}" "${pid6}"; do
  wait "${pid}" || failed=1
done

if [[ "${failed}" -ne 0 ]]; then
  echo "At least one MVSEC Dark3R evaluation failed; inspect ${OUTDIR}/mvsec_dark3r_*.log" >&2
  exit 1
fi

echo "[complete] Table5 MVSEC Dark3R evaluations"
