#!/usr/bin/env bash
set -euo pipefail

cd /data/fcr/data/Night3R

PY=/data/fcr/miniconda3/envs/stream3r/bin/python
OUTDIR=outputs/darken_cutoff_qualitative_20260716
NIGHT_CKPT=outputs/night3r_mvsec_only_adapt_1k_from_best/last_event_dark3r.pth
MAST3R_CKPT=checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth
VGGT_CKPT=/data/fcr/ckpt/model.pt
EVENTVGGT_CKPT=/data/fcr/EventVGGT/checkpoint.pt
DSEC_ROOT=/data/fcr/data/train_data/DSEC_test
MVSEC_ROOT=/data/fcr/data/train_data/MVSEC_outdoor_night2_test

mkdir -p "${OUTDIR}"

run_night3r() {
  local gpu="$1" dataset="$2" root="$3" exposure="$4"
  local config="config/eval_final_mvsec1k_dsec_darken_${exposure}_20260716.yaml"
  local output="${OUTDIR}/${dataset}_night3r_${exposure}_128.json"
  local log="${output%.json}.log"
  local -a extra=()
  if [[ "${exposure}" == "e005" ]]; then
    extra=(--prediction-dir "${OUTDIR}/qualitative/${dataset}/night3r" --prediction-count 3)
  fi
  [[ -s "${output}" ]] && { echo "[skip] ${output}"; return; }
  echo "[start] gpu=${gpu} ${dataset} Night3R ${exposure}"
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 "${PY}" utils/eval_event_dark3r_variant.py \
    --variant event_dark3r \
    --checkpoint "${NIGHT_CKPT}" \
    --config "${config}" \
    --data-root "${root}" \
    --model-name-or-path "${MAST3R_CKPT}" \
    --allow-unsafe-checkpoint-load \
    --device cuda --amp --amp-dtype bf16 \
    --batch-size 1 --max-batches 128 \
    --pair-sampling fixed --pair-stride 1 \
    --min-pair-distance 1 --max-pair-distance 1 \
    --num-workers 0 --seed 17 --event-mode real \
    "${extra[@]}" \
    --output "${output}" > "${log}" 2>&1
  echo "[done] ${output}"
}

run_vggt() {
  local gpu="$1" dataset="$2" root="$3" exposure="$4"
  local config="config/eval_final_mvsec1k_dsec_darken_${exposure}_20260716.yaml"
  local output="${OUTDIR}/${dataset}_vggt_${exposure}_128.json"
  local log="${output%.json}.log"
  local -a extra=()
  if [[ "${exposure}" == "e005" ]]; then
    extra=(--prediction-dir "${OUTDIR}/qualitative/${dataset}/vggt" --prediction-count 3)
  fi
  [[ -s "${output}" ]] && { echo "[skip] ${output}"; return; }
  echo "[start] gpu=${gpu} ${dataset} VGGT ${exposure}"
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 "${PY}" utils/eval_vggt_baseline.py \
    --variant vggt --checkpoint "${VGGT_CKPT}" \
    --config "${config}" --data-root "${root}" \
    --max-batches 128 --batch-size 1 \
    --pair-sampling fixed --pair-stride 1 \
    --min-pair-distance 1 --max-pair-distance 1 \
    --num-workers 0 --seed 17 --amp-dtype bf16 \
    "${extra[@]}" \
    --output "${output}" > "${log}" 2>&1
  echo "[done] ${output}"
}

run_eventvggt() {
  local gpu="$1" dataset="$2" root="$3"
  local config=config/eval_final_mvsec1k_dsec_darken_e005_20260716.yaml
  local output="${OUTDIR}/${dataset}_eventvggt_real_128.json"
  local log="${output%.json}.log"
  [[ -s "${output}" ]] && { echo "[skip] ${output}"; return; }
  echo "[start] gpu=${gpu} ${dataset} EventVGGT event-only"
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 "${PY}" utils/eval_vggt_baseline.py \
    --variant eventvggt --checkpoint "${EVENTVGGT_CKPT}" \
    --config "${config}" --data-root "${root}" --event-mode real \
    --max-batches 128 --batch-size 1 \
    --pair-sampling fixed --pair-stride 1 \
    --min-pair-distance 1 --max-pair-distance 1 \
    --num-workers 0 --seed 17 --amp-dtype bf16 \
    --prediction-dir "${OUTDIR}/qualitative/${dataset}/eventvggt" \
    --prediction-count 3 \
    --output "${output}" > "${log}" 2>&1
  echo "[done] ${output}"
}

(
  for exposure in e020 e010 e005; do
    run_night3r 4 dsec "${DSEC_ROOT}" "${exposure}"
  done
  run_eventvggt 4 dsec "${DSEC_ROOT}"
) & pid4=$!

(
  for exposure in e020 e010 e005; do
    run_night3r 5 mvsec "${MVSEC_ROOT}" "${exposure}"
  done
  run_eventvggt 5 mvsec "${MVSEC_ROOT}"
) & pid5=$!

(
  for exposure in e020 e010 e005; do
    run_vggt 6 dsec "${DSEC_ROOT}" "${exposure}"
  done
) & pid6=$!

(
  for exposure in e020 e010 e005; do
    run_vggt 7 mvsec "${MVSEC_ROOT}" "${exposure}"
  done
) & pid7=$!

failed=0
for pid in "${pid4}" "${pid5}" "${pid6}" "${pid7}"; do
  wait "${pid}" || failed=1
done
if [[ "${failed}" -ne 0 ]]; then
  echo "At least one darken cutoff queue failed; inspect ${OUTDIR}/*.log" >&2
  exit 1
fi

"${PY}" utils/summarize_darken_cutoff_metrics.py --input-dir "${OUTDIR}"
for dataset in dsec mvsec; do
  "${PY}" utils/render_darken_method_comparison.py \
    --prediction-root "${OUTDIR}/qualitative/${dataset}" \
    --output-dir "${OUTDIR}/qualitative/${dataset}/figures"
done
touch "${OUTDIR}/COMPLETE"
echo "[complete] ${OUTDIR}"
