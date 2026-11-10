#!/usr/bin/env bash
set -euo pipefail

cd /data/fcr/data/Night3R

PY=/data/fcr/miniconda3/envs/stream3r/bin/python
VGGT_CKPT=/data/fcr/ckpt/model.pt
EVENTVGGT_CKPT=/data/fcr/EventVGGT/checkpoint.pt
OUTDIR=outputs/vggt_eventvggt_darken_fixed128_20260716
DSEC_ROOT=/data/fcr/data/train_data/DSEC_test
MVSEC_ROOT=/data/fcr/data/train_data/MVSEC_outdoor_night2_test
EVENT_CONFIG=config/train_night3r_mvsec_only_adapt_1k_from_best.yaml

mkdir -p "${OUTDIR}"

run_vggt() {
  local gpu="$1"
  local dataset="$2"
  local root="$3"
  local exposure="$4"
  local config="config/eval_final_mvsec1k_dsec_darken_${exposure}_20260716.yaml"
  local output="${OUTDIR}/${dataset}_vggt_${exposure}_128.json"
  local log="${output%.json}.log"
  if [[ -s "${output}" ]]; then
    echo "[skip] ${output}"
    return
  fi
  echo "[start] gpu=${gpu} dataset=${dataset} method=vggt exposure=${exposure}"
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 "${PY}" utils/eval_vggt_baseline.py \
    --variant vggt \
    --checkpoint "${VGGT_CKPT}" \
    --config "${config}" \
    --data-root "${root}" \
    --max-batches 128 \
    --batch-size 1 \
    --pair-sampling fixed \
    --pair-stride 1 \
    --min-pair-distance 1 \
    --max-pair-distance 1 \
    --num-workers 0 \
    --seed 17 \
    --amp-dtype bf16 \
    --output "${output}" \
    > "${log}" 2>&1
  echo "[done] ${output}"
}

run_eventvggt() {
  local gpu="$1"
  local dataset="$2"
  local root="$3"
  local output="${OUTDIR}/${dataset}_eventvggt_real_128.json"
  local log="${output%.json}.log"
  if [[ -s "${output}" ]]; then
    echo "[skip] ${output}"
    return
  fi
  echo "[start] gpu=${gpu} dataset=${dataset} method=eventvggt event=real"
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 "${PY}" utils/eval_vggt_baseline.py \
    --variant eventvggt \
    --checkpoint "${EVENTVGGT_CKPT}" \
    --config "${EVENT_CONFIG}" \
    --data-root "${root}" \
    --event-mode real \
    --max-batches 128 \
    --batch-size 1 \
    --pair-sampling fixed \
    --pair-stride 1 \
    --min-pair-distance 1 \
    --max-pair-distance 1 \
    --num-workers 0 \
    --seed 17 \
    --amp-dtype bf16 \
    --output "${output}" \
    > "${log}" 2>&1
  echo "[done] ${output}"
}

(
  run_vggt 4 dsec "${DSEC_ROOT}" e020
  run_vggt 4 mvsec "${MVSEC_ROOT}" e020
) &
pid4=$!

(
  run_vggt 5 dsec "${DSEC_ROOT}" e010
  run_vggt 5 mvsec "${MVSEC_ROOT}" e010
) &
pid5=$!

(
  run_vggt 6 dsec "${DSEC_ROOT}" e005
  run_eventvggt 6 dsec "${DSEC_ROOT}"
) &
pid6=$!

(
  run_vggt 7 mvsec "${MVSEC_ROOT}" e005
  run_eventvggt 7 mvsec "${MVSEC_ROOT}"
) &
pid7=$!

failed=0
for pid in "${pid4}" "${pid5}" "${pid6}" "${pid7}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done

if [[ "${failed}" -ne 0 ]]; then
  echo "One or more VGGT/EventVGGT queues failed; inspect ${OUTDIR}/*.log" >&2
  exit 1
fi

"${PY}" utils/summarize_vggt_eventvggt_darken.py --input-dir "${OUTDIR}"
touch "${OUTDIR}/COMPLETE"
echo "[complete] ${OUTDIR}"
